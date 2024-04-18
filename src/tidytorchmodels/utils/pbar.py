# https://stackoverflow.com/questions/6793748/python-doing-conditional-imports-the-right-way
try:
    from tqdm.auto import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        class tqdm(object):
            """A simpler placeholder that prints progress on the same line, in case tqdm is not installed.
            Adapted from https://github.com/timesler/facenet-pytorch/blob/v2.4.1/models/utils/download.py
            Will print progress in MB for file downloads and just in iterations for everything else.
            """
            def __init__(self, total=None, unit=None, unit_scale=None, unit_divisor=None):
                self.n = 0
                self.b = unit == 'B'
                self.total = total
                if total and self.b:
                    self.total /= 1024 ** 2

            def update(self, n):
                if not self.b:
                    self.n += n
                    units = ''
                else:
                    self.n += int(n / 1024 ** 2)
                    units = 'MB'
                if self.total is None:
                    print('\r%d%s' % (self.n, units), end='')
                else:
                    percentage = int(100. * self.n / self.total + 0.5)
                    print('\r%d/%d%s (%d%%)' % (self.n, self.total, units, percentage), end='')

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                print('\r')