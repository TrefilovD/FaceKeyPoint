import argparse
import os
import os.path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_points(dir_path, max_points):
  print('Reading directory {}'.format(dir_path))
  points = {}
  i = 0
  for idx, fname in enumerate(os.listdir(dir_path)):
    if max_points is not None and idx > max_points:
      break

    cur_path = os.path.join(dir_path, fname)
    # TODO: add ability to exclude path
    #if os.path.isdir(cur_path):
    #  points.update(read_points(cur_path, max_points))

    if cur_path.endswith('.pts') or cur_path.endswith('.pts1'):
      if idx % 100 == 0:
        print(idx)

      with open(cur_path) as cur_file:
        lines = cur_file.readlines()
        if lines[0].startswith('version'): # to support different formats
          lines = lines[3:-1]
        mat = np.fromstring(''.join(lines), sep=' ')
        points[fname] = (mat[0::2], mat[1::2])

  return points


def count_ced(predicted_points, gt_points, normalization_type, left_eye_idx, right_eye_idx):
  ceds = defaultdict(list)

  if normalization_type == 'eyes':
    left_eye_idx = left_eye_idx.split(',')
    right_eye_idx = right_eye_idx.split(',')
    assert (len(left_eye_idx) == 1 and len(right_eye_idx) == 1) or \
            (len(left_eye_idx) == 2 and len(right_eye_idx) == 2)

  for method_name in predicted_points.keys():
    for img_name in predicted_points[method_name].keys():
      if img_name in gt_points:
        #print('Processing key {}'.format(img_name))
        x_pred, y_pred = predicted_points[method_name][img_name]
        x_gt, y_gt = gt_points[img_name]
        n_points = x_pred.shape[0]
        assert n_points == x_gt.shape[0], '{} != {}'.format(n_points, x_gt.shape[0])

        if normalization_type == 'eyes':
          x_left_eye = np.mean([x_gt[int(idx)] for idx in left_eye_idx])
          x_right_eye = np.mean([x_gt[int(idx)] for idx in right_eye_idx])
          y_left_eye = np.mean([y_gt[int(idx)] for idx in left_eye_idx])
          y_right_eye = np.mean([y_gt[int(idx)] for idx in right_eye_idx])

          normalization_factor = np.linalg.norm([x_left_eye - x_right_eye, y_left_eye - y_right_eye])
        elif normalization_type == 'bbox':
          w = np.max(x_gt) - np.min(x_gt)
          h = np.max(y_gt) - np.min(y_gt)
          normalization_factor = np.sqrt(h * w)
        else:
          raise Exception('Wrong normalization type')

        diff_x = [x_gt[i] - x_pred[i] for i in range(n_points)]
        diff_y = [y_gt[i] - y_pred[i] for i in range(n_points)]
        dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
        avg_norm_dist = np.sum(dist) / (n_points * normalization_factor)
        ceds[method_name].append(avg_norm_dist)
        #print('Average distance for method {} = {}'.format(method_name, avg_norm_dist))
      else:
        print('Skipping key {}, because its not in the gt points'.format(img_name))
    ceds[method_name] = np.sort(ceds[method_name])

  return ceds


def count_ced_auc(errors):
  if not isinstance(errors, list):
    errors = [errors]

  aucs = []
  for error in errors:
    auc = 0
    proportions = np.arange(error.shape[0], dtype=np.float32) / error.shape[0]
    assert (len(proportions) > 0)

    step = 0.01
    for thr in np.arange(0.0, 1.0, step):
      gt_indexes = [idx for idx, e in enumerate(error) if e >= thr]
      if len(gt_indexes) > 0:
        first_gt_idx = gt_indexes[0]
      else:
        first_gt_idx = len(error) - 1
      auc += proportions[first_gt_idx] * step
    aucs.append(auc)
  return aucs


def main():
  parser = argparse.ArgumentParser(description='CED computation script',
                                   add_help=True)
  parser.add_argument('--gt_path', action='store', type=str, help='')
  parser.add_argument('--predictions_path', action='append', type=str, help='')
  parser.add_argument('--output_path', action='store', type=str, help='')
  parser.add_argument('--left_eye_idx', action='store', type=str, help='')
  parser.add_argument('--right_eye_idx', action='store', type=str, help='')
  parser.add_argument('--normalization_type', action='store', type=str, help='',
                      choices=['bbox', 'eyes'], required=True)
  parser.add_argument('--max_points_to_read', action='store', type=int, help='',
                      default=None)
    
  parser.add_argument('--error_thr', action='store', type=float, help='',
                      default=0.08)
  parser.add_argument("--val", action="store_true", type=bool)
  args = parser.parse_args()
    
  print('args.error_thr = {}'.format(args.error_thr))
  
  predicted_points = {}
  for pred_path in args.predictions_path:
    predicted_points[os.path.basename(pred_path)] = read_points(pred_path, args.max_points_to_read) # {"test": {"img_name": (np(X), np(Y)), ...}}
  gt_points = read_points(args.gt_path, args.max_points_to_read) # {"img_name": (np(X), np(Y)), ...}
  #print(predicted_points.keys())
  #print(gt_points)

  ceds = count_ced(predicted_points, gt_points, args)

  # saving figure
  line_styles = [':', '-.', '--', '-']
  plt.figure(figsize=(30,20), dpi=100)
  for method_idx, method_name in enumerate(ceds.keys()):
    print('Plotting graph for the method {}'.format(method_name))
    err = ceds[method_name]
    proportion = np.arange(err.shape[0], dtype=np.float32) / err.shape[0]
    under_thr = err > args.error_thr
    last_idx = len(err)
    if len(np.flatnonzero(under_thr)) > 0:
      last_idx = np.flatnonzero(under_thr)[0]
    under_thr_range = range(last_idx)
    cur_auc = count_ced_auc(err)[0]
    
    plt.plot(err[under_thr_range], proportion[under_thr_range], label=method_name + ', auc={:1.3f}'.format(cur_auc),
	     linestyle=line_styles[method_idx % len(line_styles)], linewidth=2.0)
  plt.legend(loc='right', prop={'size': 24})
  plt.savefig(args.output_path)


def val_ced_calculate(predicted_points, gt_points, output_path, normalization_type="eyes", error_thr=0.08, left_eye_idx="0", right_eye_idx='1', epoch = -1):
  ceds = count_ced(predicted_points, gt_points, normalization_type, left_eye_idx, right_eye_idx)
  fig, ax = plt.subplots(len(ceds.keys()), 1, figsize=(30,20), dpi=100)
  # saving figure
  line_styles = ['-']
  # plt.figure(figsize=(30,20), dpi=100)
  for method_idx, method_name in enumerate(ceds.keys()):
    err = ceds[method_name]
    proportion = np.arange(err.shape[0], dtype=np.float32) / err.shape[0]
    under_thr = err > error_thr
    last_idx = len(err)
    if len(np.flatnonzero(under_thr)) > 0:
      last_idx = np.flatnonzero(under_thr)[0]
    under_thr_range = range(last_idx)
    cur_auc = count_ced_auc(err)[0]

    if len(ceds.keys()) > 1:
      ax[method_idx].plot(err[under_thr_range], proportion[under_thr_range], label=method_name + ', auc={:1.3f}'.format(cur_auc),
	     linestyle=line_styles[method_idx % len(line_styles)], linewidth=2.0)
      ax[method_idx].set_title(method_name)
    else:
      ax.plot(err[under_thr_range], proportion[under_thr_range], label=method_name + ', auc={:1.3f}'.format(cur_auc),
	     linestyle=line_styles[method_idx % len(line_styles)], linewidth=2.0)
      ax.set_title(method_name)
  plt.legend(loc='right', prop={'size': 24})
  plt.xlabel(f"error@{error_thr}")
  plt.ylabel("CED")
  plt.title(f"CED {'epoch' + str(epoch) if epoch != -1 else ''}")
  return fig


if __name__ == '__main__':
  main()
