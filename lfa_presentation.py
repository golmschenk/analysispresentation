import os
import re

import matplotlib2tikz
from matplotlib import ticker
from log import Log
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('module://backend_interagg')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('deep', n_colors=9)

logs_directory = 'loss function analysis logs'
log_paths = Log.find_log_paths(logs_directory)


@ticker.FuncFormatter
def human_number_format(x, pos):
    """Formats to a human understanding suffix."""
    magnitude = 0
    while abs(x) >= 1000:
        magnitude += 1
        x /= 1000.0
    return f'{x:g}{["", "K", "M", "G", "T", "P"][magnitude]}'


def fix_tick_label_display(tikz_path):
    with open(tikz_path, 'r') as file:
        file_contents = file.read()
    file_contents = file_contents.replace('xmajorticks=false,', 'xtick style={draw=none},')
    file_contents = file_contents.replace('ymajorticks=false,', 'ytick style={draw=none},')
    with open(tikz_path, 'w') as file:
        file.write(file_contents)


def add_tikz_axis_option(tikz_path, option):
    with open(tikz_path, 'r') as file:
        file_lines = file.readlines()
    axis_start_index = file_lines.index('\\begin{axis}[\n')
    file_lines.insert(axis_start_index + 1, option + ',\n')
    with open(tikz_path, 'w') as file:
        file.writelines(file_lines)


def remove_tikz_axis_option(tikz_path, option):
    with open(tikz_path, 'r') as file:
        file_lines = file.readlines()
    file_lines.remove(option + ',\n')
    with open(tikz_path, 'w') as file:
        file.writelines(file_lines)


long_log_paths = [path for path in log_paths if 'lfa long' in path and 'GAN' in path]
for long_log_path in long_log_paths:
    match = re.search(r'(.*) long(.* )ul1e(-?\d) fl1e(-?\d)( .*/GAN).*', long_log_path)
    long_ul_exponent = int(match.group(3))
    long_fl_exponent = int(match.group(4))
    figure, axes = plt.subplots(dpi=300)
    matching_function = re.search(r' (abs_mean|square_mean|norm_mean) ', long_log_path).group(1)
    contrasting_function = re.search(r' (abs_mean_neg|abs_plus_one_log_mean_neg|abs_plus_one_sqrt_mean_neg) ',
                                     long_log_path).group(1)
    for ul_exponent in [long_ul_exponent-1, long_ul_exponent, long_ul_exponent+1]:
        for fl_exponent in [long_fl_exponent - 1, long_fl_exponent, long_fl_exponent + 1]:
            log = Log(f'{match.group(1)}{match.group(2)}ul1e{ul_exponent} fl1e{fl_exponent}{match.group(5)}')
            log.scalars_data_frame = log.scalars_data_frame.drop(
                log.scalars_data_frame[(log.scalars_data_frame['Step'] % 5000 != 0)
                                       & ((log.scalars_data_frame['Step'] + 1) % 5000 != 0)].index)
            log.scalars_data_frame['Step/1000'] = log.scalars_data_frame['Step'] / 1000
            log.scalars_data_frame.plot(x='Step/1000', y='1_Validation_Error/MAE', ax=axes,
                                        label=f'M 1e{ul_exponent}, C 1e{fl_exponent}')
    matching_function_short = {'abs_mean': '$f_{abs}$', 'square_mean': '$f_{square}$',
                               'norm_mean': '$f_{norm}$'}[matching_function]
    contrasting_function_short = {'abs_mean_neg': '$f_{-abs}$', 'abs_plus_one_log_mean_neg': '$f_{-log}$',
                                  'abs_plus_one_sqrt_mean_neg': '$f_{-sqrt}$'}[contrasting_function]
    axes.set_title(f'Matching: {matching_function_short}\\\\Contrasting: {contrasting_function_short}')
    axes.set_xlabel('Training Step (K)')
    axes.set_ylabel('MAE')
    axes.set_ylim([0.05, 0.3])
    axes.set_xlim([0, 100])
    axes.legend(ncol=2)
    axes.xaxis.set_major_formatter(human_number_format)
    os.makedirs('latex', exist_ok=True)
    tikz_path = os.path.join('latex', f'matching-{matching_function.replace("_", "-")}-' +
                                      f'contrasting-{contrasting_function.replace("_", "-")}.tikz')
    matplotlib2tikz.save(tikz_path)
    fix_tick_label_display(tikz_path)
    add_tikz_axis_option(tikz_path, 'xticklabel style={/pgf/number format/fixed}')
    add_tikz_axis_option(tikz_path, 'yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2}')
    add_tikz_axis_option(tikz_path, r'legend style={font=\tiny}')
    add_tikz_axis_option(tikz_path, r'font=\scriptsize')
    add_tikz_axis_option(tikz_path, r'legend style={at={(0.5,-0.36)}, anchor=north, draw=none, fill=none}')
    add_tikz_axis_option(tikz_path, r'title style={align=left}')
    add_tikz_axis_option(tikz_path, 'enlargelimits=false')
    remove_tikz_axis_option(tikz_path, 'legend style={draw=white!80.0!black}')
    figure.show()
    plt.close(figure)
