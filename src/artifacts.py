"""Save report tables and figures, grouped by assignment and experiment."""
from pathlib import Path

import pandas as pd


ASSIGNMENTS = {'lenet': 1, 'googlenet': 2, 'resnet18': 3, 'resnet50': 4}


def default_output_root():
    kaggle = Path('/kaggle/working')
    if kaggle.is_dir():
        return kaggle / 'outputs'
    return Path(__file__).resolve().parents[1] / 'outputs'


def export_result(result, output_root=None):
    root = Path(output_root) if output_root is not None else default_output_root()
    summary = result['summary']
    assignment = ASSIGNMENTS[result['config']['model']]
    folder = root / f'bai{assignment}' / summary['experiment']
    tables, figures = folder / 'tables', folder / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary]).to_csv(tables / 'metrics.csv', index=False, encoding='utf-8-sig')
    result['history'].to_csv(tables / 'history.csv', index=False, encoding='utf-8-sig')
    for name, table in result['model_tables'].items():
        filename = 'inception_blocks.csv' if name == 'inception' else 'layer_table.csv'
        table.to_csv(tables / filename, index=False, encoding='utf-8-sig')
    (figures / 'loss_curve.png').write_bytes(result['loss_image'])

    evaluation = result['evaluation']
    if evaluation is not None:
        evaluation['per_class'].to_csv(tables / 'per_class_scores.csv', encoding='utf-8-sig')
        pd.DataFrame(evaluation['confusion_matrix'], index=result['classes'],
                     columns=result['classes']).to_csv(
            tables / 'confusion_matrix.csv', index_label='true_class', encoding='utf-8-sig')
        (figures / 'confusion_matrix.png').write_bytes(evaluation['confusion_image'])
    else:
        # A validation-only rerun must not leave test artifacts from an older run.
        for path in (tables / 'per_class_scores.csv', tables / 'confusion_matrix.csv',
                     figures / 'confusion_matrix.png'):
            path.unlink(missing_ok=True)
    return folder


def export_comparison(table, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding='utf-8-sig')
    return path
