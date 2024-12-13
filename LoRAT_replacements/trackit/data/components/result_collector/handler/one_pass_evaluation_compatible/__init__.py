import os.path
from typing import Optional, Sequence, Mapping, Tuple, MutableMapping

from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from ..utils.writer import FolderWriter, ZipfileWriter, PlainFolderWriter
from ...progress_tracer import EvaluationProgress
from .. import EvaluationResultHandler
from .ope_metrics import OPEMetrics, DatasetOPEMetricsListBuilder, DatasetOPEMetricsList, compute_OPE_metrics_mean, compute_one_pass_evaluation_metrics
from .report_gen import generate_dataset_one_pass_evaluation_report, generate_one_pass_evaluation_summary_report, \
    dump_sequence_tracking_results_with_groundtruth, generate_sequence_one_pass_evaluation_report
from ..utils.compatibility import ExternalToolkitCompatibilityHelper


class EvaluationResultPersistenceWithOPEMetricsCompatibleWithExternalTools(EvaluationResultHandler):
    def __init__(self, tracker_name: str, output_path: Optional[str], file_name: Optional[str], rasterize_bbox: bool):
        self._compliance = 'STARK'  # todo: options to match with evaluation results on pytracking, LaSOT toolkit, OTB toolkit
        self._tracker_name = tracker_name
        self._folder_writer = None
        if output_path is not None and file_name is not None:
            self._folder_writer = PlainFolderWriter(os.path.join(output_path, file_name))
            # self._folder_writer = ZipfileWriter(os.path.join(output_path, file_name + '.zip'))

        self._progress_aware_sub_handler = EvaluationResultPersistenceWithOPEMetrics_ProgressAware(rasterize_bbox)
        self._live_feed_sub_handler = EvaluationResultPersistenceWithOPEMetrics_LiveFeed(rasterize_bbox)
        self._collected_metrics = []
        self._is_closed = False

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT],
               evaluation_progresses: Sequence[EvaluationProgress]):
        assert not self._is_closed
        metrics = {}
        sub_handler_1_metrics = self._progress_aware_sub_handler(self._tracker_name, self._folder_writer, evaluation_results, evaluation_progresses)
        sub_handler_2_metrics = self._live_feed_sub_handler(self._tracker_name, self._folder_writer, evaluation_results, evaluation_progresses)
        if sub_handler_1_metrics is not None:
            metrics.update(sub_handler_1_metrics)
        if sub_handler_2_metrics is not None:
            metrics.update(sub_handler_2_metrics)
        if len(metrics) != 0:
            self._collected_metrics.extend(list(metrics.items()))

    def close(self):
        assert not self._is_closed
        metrics = {}
        sub_handler_1_metrics = self._progress_aware_sub_handler.finalize(self._tracker_name, self._folder_writer)
        sub_handler_2_metrics = self._live_feed_sub_handler.finalize(self._tracker_name, self._folder_writer)
        if sub_handler_1_metrics is not None:
            metrics.update(sub_handler_1_metrics)
        if sub_handler_2_metrics is not None:
            metrics.update(sub_handler_2_metrics)
        if len(metrics) != 0:
            self._collected_metrics.extend(list(metrics.items()))
        if self._folder_writer is not None:
            self._folder_writer.close()
            self._folder_writer = None
        self._is_closed = True

    def get_metrics(self) -> Optional[Sequence[Tuple[str, float]]]:
        return self._collected_metrics


class FinalOPEMetricsSummaryReportGenerator:
    def __init__(self):
        self._final_summary_metrics = {}

    def add(self, dataset_name: str, repeat_index: Optional[int], metrics: OPEMetrics):
        if repeat_index not in self._final_summary_metrics:
            self._final_summary_metrics[repeat_index] = {}
        this_repeat_summary_metrics = self._final_summary_metrics[repeat_index]
        assert dataset_name not in this_repeat_summary_metrics
        this_repeat_summary_metrics[dataset_name] = metrics

    def dump(self, folder_writer: FolderWriter, tracker_name: str):
        for repeat_index, this_repeat_summary_metrics in self._final_summary_metrics.items():
            sorted_metrics = dict(sorted(this_repeat_summary_metrics.items()))
            generate_one_pass_evaluation_summary_report(folder_writer, tracker_name, repeat_index, sorted_metrics)


def __get_summary_metric_name(metric_name: str, dataset_name: str, repeat_index: Optional[int]):
    if repeat_index is None:
        return f'{metric_name}_{dataset_name}'
    else:
        return f'{metric_name}_{dataset_name}_{repeat_index:03d}'


def _generate_dataset_summary_metrics_name_value_pair(dataset_name: str, repeat_index: Optional[int], metrics: OPEMetrics):
    return {
        __get_summary_metric_name('success_score', dataset_name, repeat_index): metrics.success_score,
        __get_summary_metric_name('precision_score', dataset_name, repeat_index): metrics.precision_score,
        __get_summary_metric_name('norm_precision_score', dataset_name, repeat_index): metrics.normalized_precision_score,
        __get_summary_metric_name('success_rate_at_overlap_0_5', dataset_name, repeat_index): metrics.success_rate_at_overlap_0_5,
        __get_summary_metric_name('success_rate_at_overlap_0_75', dataset_name, repeat_index): metrics.success_rate_at_overlap_0_75,
        __get_summary_metric_name('fps', dataset_name, repeat_index): metrics.get_fps(),
    }


class EvaluationResultPersistenceWithOPEMetrics_ProgressAware:
    def __init__(self, rasterize_bbox: bool):
        self._known_tracks_metric_cache = {}
        self._multi_run_dataset_metrics_cache = {}
        self._final_summary_report_generator = FinalOPEMetricsSummaryReportGenerator()
        self._compatibility_helper = ExternalToolkitCompatibilityHelper()
        self._rasterize_bbox = rasterize_bbox

    def __call__(self, tracker_name: str, folder_writer: Optional[FolderWriter],
                 evaluation_results: Sequence[SequenceEvaluationResult_SOT],
                 evaluation_progresses: Sequence[EvaluationProgress]) -> Optional[Mapping[str, float]]:
        summary_metrics = {}

        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            if evaluation_progress.this_dataset is None:
                continue

            assert evaluation_result.sequence_info.dataset_full_name is not None
            assert evaluation_result.sequence_info.sequence_name is not None
            predicted_target_bounding_boxes = evaluation_result.output_box
            if self._rasterize_bbox:
                predicted_target_bounding_boxes = bbox_rasterize(predicted_target_bounding_boxes)

            metrics, frames_iou = \
                compute_one_pass_evaluation_metrics(evaluation_result.sequence_info.dataset_name,
                                                    predicted_target_bounding_boxes,
                                                    evaluation_result.groundtruth_box,
                                                    evaluation_result.groundtruth_object_existence_flag,
                                                    evaluation_result.time_cost,
                                                    self._compatibility_helper)

            print(f'{evaluation_result.sequence_info.sequence_name}: success {metrics.success_score:.04f}, prec {metrics.precision_score:.04f}, norm_pre {metrics.normalized_precision_score:.04f}')

            repeat_index = evaluation_progress.repeat_index
            if evaluation_progress.this_dataset.total_repeat_times == 1:
                repeat_index = None
            cache_key = evaluation_result.sequence_info.dataset_full_name, repeat_index
            if cache_key not in self._known_tracks_metric_cache:
                dataset_metrics_list_builder = DatasetOPEMetricsListBuilder()
                self._known_tracks_metric_cache[cache_key] = dataset_metrics_list_builder
            else:
                dataset_metrics_list_builder = self._known_tracks_metric_cache[cache_key]
            dataset_metrics_list_builder.append(evaluation_result.sequence_info.sequence_name, metrics)
            if folder_writer is not None:
                dump_sequence_tracking_results_with_groundtruth(folder_writer,
                                                                tracker_name, repeat_index,
                                                                evaluation_result.sequence_info.dataset_full_name,
                                                                evaluation_result.sequence_info.sequence_name,
                                                                evaluation_result.evaluated_frame_indices,
                                                                evaluation_result.output_confidence,
                                                                predicted_target_bounding_boxes,
                                                                evaluation_result.groundtruth_object_existence_flag,
                                                                evaluation_result.groundtruth_box,
                                                                evaluation_result.time_cost,
                                                                frames_iou)
                generate_sequence_one_pass_evaluation_report(folder_writer,
                                                             tracker_name, repeat_index,
                                                             evaluation_result.sequence_info.dataset_full_name,
                                                             evaluation_result.sequence_info.sequence_name,
                                                             metrics)
            if evaluation_progress.this_dataset.this_repeat_all_evaluated:
                dataset_metrics_list = dataset_metrics_list_builder.build()
                dataset_metrics_list = dataset_metrics_list.sort_by_sequence_name()

                del self._known_tracks_metric_cache[cache_key]

                dataset_summary_metrics = dataset_metrics_list.get_mean()
                if folder_writer is not None:
                    generate_dataset_one_pass_evaluation_report(folder_writer,
                                                                tracker_name, repeat_index,
                                                                evaluation_result.sequence_info.dataset_full_name,
                                                                dataset_metrics_list,
                                                                dataset_summary_metrics)
                self._final_summary_report_generator.add(evaluation_result.sequence_info.dataset_full_name, repeat_index, dataset_summary_metrics)
                summary_metrics.update(
                    _generate_dataset_summary_metrics_name_value_pair(evaluation_result.sequence_info.dataset_full_name,
                                                                      repeat_index,
                                                                      dataset_summary_metrics))
                if evaluation_progress.this_dataset.total_repeat_times > 1:
                    if evaluation_result.sequence_info.dataset_full_name not in self._multi_run_dataset_metrics_cache:
                        dataset_all_runs_metrics = []
                        self._multi_run_dataset_metrics_cache[evaluation_result.sequence_info.dataset_full_name] = dataset_all_runs_metrics
                    else:
                        dataset_all_runs_metrics = self._multi_run_dataset_metrics_cache[evaluation_result.sequence_info.dataset_full_name]
                    dataset_all_runs_metrics.append(dataset_summary_metrics)

                    if evaluation_progress.this_dataset.all_evaluated:
                        dataset_all_runs_mean_metrics = compute_OPE_metrics_mean(dataset_all_runs_metrics)
                        del self._multi_run_dataset_metrics_cache[evaluation_result.sequence_info.dataset_full_name]
                        summary_metrics.update(_generate_dataset_summary_metrics_name_value_pair(evaluation_result.sequence_info.dataset_full_name, None, dataset_all_runs_mean_metrics))
                        self._final_summary_report_generator.add(evaluation_result.sequence_info.dataset_full_name, None, dataset_all_runs_mean_metrics)

        return summary_metrics

    def finalize(self, tracker_name: str, folder_writer: Optional[FolderWriter]) -> Optional[Mapping[str, float]]:
        if folder_writer is not None:
            self._final_summary_report_generator.dump(folder_writer, tracker_name)

        return None


class EvaluationResultPersistenceWithOPEMetrics_LiveFeed:
    def __init__(self, rasterize_bbox: bool):
        self._metric_cache: MutableMapping[Tuple[str, int], DatasetOPEMetricsListBuilder] = {}
        self._compatibility_helper = ExternalToolkitCompatibilityHelper()
        self._rasterize_bbox = rasterize_bbox

    def __call__(self,
                 tracker_name: str, folder_writer: Optional[FolderWriter],
                 evaluation_results: Sequence[SequenceEvaluationResult_SOT],
                 evaluation_progresses: Sequence[EvaluationProgress]) -> Optional[Mapping[str, float]]:
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            if evaluation_progress.this_dataset is not None:
                continue
            assert evaluation_result.sequence_info.dataset_full_name is not None
            assert evaluation_result.sequence_info.sequence_name is not None
            predicted_target_bounding_boxes = evaluation_result.output.box
            metrics, frames_iou = \
                compute_one_pass_evaluation_metrics(evaluation_result.sequence_info.dataset_name,
                                                    predicted_target_bounding_boxes,
                                                    evaluation_result.groundtruth_box,
                                                    evaluation_result.groundtruth_object_existence_flag,
                                                    evaluation_result.time_cost,
                                                    self._compatibility_helper)

            if folder_writer is not None:
                dump_sequence_tracking_results_with_groundtruth(folder_writer,
                                                                tracker_name, evaluation_progress.repeat_index,
                                                                evaluation_result.sequence_info.dataset_full_name,
                                                                evaluation_result.sequence_info.sequence_name,
                                                                evaluation_result.evaluated_frame_indices,
                                                                evaluation_result.output_confidence,
                                                                predicted_target_bounding_boxes,
                                                                evaluation_result.groundtruth_box,
                                                                evaluation_result.groundtruth_object_existence_flag,
                                                                evaluation_result.time_cost,
                                                                frames_iou)
                generate_sequence_one_pass_evaluation_report(folder_writer,
                                                             tracker_name, evaluation_progress.repeat_index,
                                                             evaluation_result.sequence_info.dataset_full_name,
                                                             evaluation_result.sequence_info.sequence_name,
                                                             metrics)
            metric_cache_key = evaluation_result.sequence_info.dataset_full_name, evaluation_progress.repeat_index
            if metric_cache_key not in self._metric_cache:
                self._metric_cache[metric_cache_key] = DatasetOPEMetricsListBuilder()
            self._metric_cache[metric_cache_key].append(evaluation_result.sequence_info.sequence_name, metrics)
        return None

    def finalize(self, tracker_name: str, folder_writer: Optional[FolderWriter]) -> Optional[Mapping[str, float]]:
        summary_metrics = {}

        all_dataset_summary_metrics = {}

        for (dataset_full_name, repeat_index), metrics_list_builder in self._metric_cache.items():
            metrics_list = metrics_list_builder.build()
            dataset_summary_metrics = metrics_list.get_mean()
            if folder_writer is not None:
                generate_dataset_one_pass_evaluation_report(folder_writer, tracker_name, repeat_index,
                                                            dataset_full_name, metrics_list.sort_by_sequence_name(),
                                                            dataset_summary_metrics)
            if dataset_full_name not in all_dataset_summary_metrics:
                all_dataset_summary_metrics[dataset_full_name] = []
            all_dataset_summary_metrics[dataset_full_name].append(dataset_summary_metrics)

        final_summary_report_generator = FinalOPEMetricsSummaryReportGenerator()

        for dataset_full_name, dataset_summary_metrics_list in all_dataset_summary_metrics.items():
            for repeat_index, dataset_summary_metrics in enumerate(dataset_summary_metrics_list):
                summary_metrics.update(_generate_dataset_summary_metrics_name_value_pair(dataset_full_name, repeat_index, dataset_summary_metrics))
                final_summary_report_generator.add(dataset_full_name, repeat_index, dataset_summary_metrics)
            if len(dataset_summary_metrics_list) > 1:
                dataset_multirun_averaged_metrics = compute_OPE_metrics_mean(dataset_summary_metrics_list)
            else:
                dataset_multirun_averaged_metrics = dataset_summary_metrics_list[0]
            summary_metrics.update(_generate_dataset_summary_metrics_name_value_pair(dataset_full_name, None, dataset_multirun_averaged_metrics))
            final_summary_report_generator.add(dataset_full_name, None, dataset_multirun_averaged_metrics)

        if folder_writer is not None:
            final_summary_report_generator.dump(folder_writer, tracker_name)

        return summary_metrics