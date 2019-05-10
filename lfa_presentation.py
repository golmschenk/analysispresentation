
import re
from matplotlib import ticker
from log import Log
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('module://backend_interagg')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('deep', n_colors=9)

logs_directory = 'logs'
log_paths = Log.find_log_paths(logs_directory)


@ticker.FuncFormatter
def human_number_format(x, pos):
    """Formats to a human understanding suffix."""
    magnitude = 0
    while abs(x) >= 1000:
        magnitude += 1
        x /= 1000.0
    return f'{x:g}{["", "K", "M", "G", "T", "P"][magnitude]}'


long_log_paths = [path for path in log_paths if 'lfa long' in path and 'GAN' in path]
for long_log_path in long_log_paths:
    match = re.search(r'(.*) long(.* )ul1e(-?\d) fl1e(-?\d)( .*/GAN).*', long_log_path)
    long_ul_exponent = int(match.group(3))
    long_fl_exponent = int(match.group(4))
    figure, axes = plt.subplots(dpi=300)
    matching_function = re.search(r' (abs_mean|square_mean|norm_mean) ', long_log_path).group(1)
    contrasting_function = re.search(r' (abs_mean_neg|abs_plus_one_log_mean_neg|abs_plus_one_sqrt_mean_neg) ',
                                     long_log_path).group(1)
    axes.set_title(f'{matching_function} {contrasting_function}')
    axes.set_xlabel('Training Step')
    axes.set_ylabel('MAE')
    for ul_exponent in [long_ul_exponent-1, long_ul_exponent, long_ul_exponent+1]:
        for fl_exponent in [long_fl_exponent - 1, long_fl_exponent, long_fl_exponent + 1]:
            log = Log(f'{match.group(1)}{match.group(2)}ul1e{ul_exponent} fl1e{fl_exponent}{match.group(5)}')
            log.scalars_data_frame = log.scalars_data_frame.drop(
                log.scalars_data_frame[(log.scalars_data_frame['Step'] % 5000 != 0)
                                       & ((log.scalars_data_frame['Step'] + 1) % 5000 != 0)].index)
            log.scalars_data_frame.plot(y='1_Validation_Error/MAE', ax=axes,
                                        label=f'UL 1e{ul_exponent} FL 1e{fl_exponent}')
    axes.set_ylim([0.05, 0.3])
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes.xaxis.set_major_formatter(human_number_format)
    figure.show()
    plt.close(figure)



# For each.
    # Get its loss exponents.
    # Get the surrounding loss exponents.
    # Get each short log with matching exponents.
    # Plot the short logs together.

# After
    # Smooth values.
    # Make all plots have the same range.