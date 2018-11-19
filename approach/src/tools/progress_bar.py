import sys


def print_progressbar(count, total, suffix=''):
    if not total:
        sys.stdout.write('processed issues: %s\r' % count)
        sys.stdout.flush()
        return

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()
