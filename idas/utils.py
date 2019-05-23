from tensorflow.python.client import device_lib
import time
import sys
import os


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.makedirs(path)
    except OSError:
        pass


def create_gif(gif_name, path, ext='.png', delay=30, loop=0):
    """ Create gif from the list of images under the given path
    On Mac OS X, it requires "brew install ImageMagick".
    """
    if path[-1] == '/':
        path = path[:-1]
    if gif_name[-4:] == '.gif':
        gif_name = gif_name[:-4]

    cmd = 'convert -delay {0} -loop {1} {2}/*{3} {2}/{4}.gif'.format(delay, loop, path, ext, gif_name)
    os.system(cmd)
    print('gif successfully saved as {0}.gif'.format(os.path.join(path, gif_name)))


def print_yellow_text(text, sep=True):
    """ useful for debug """
    if sep:
        print('_' * 40)  # line separator
    print('\033[1;33m{0}\033[0m'.format(text))


class bcolors:
    """ Colors for formatted text.
    Example: print(bcolors.WARNING + "Warning: This is a warning." + bcolors.ENDC)
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ProgressBar(object):
    def __init__(self, update_delay):
        """
        Progress bar.
        :param update_delay: (int) time to wait before updating the progress bar (in seconds).

        ----------
        Example 1:

            bar = ProgressBar(0.2)
            bar.attach()
            for i in range(100):
                bar.monitor_progress()
                time.sleep2(0.1)
            bar.detach()

        Example 2: monitor also ETA (Estimated Time of Arrival)
                    notice that the progress bar is detached first and than the lta is updated

            bar = ProgressBar(0.5)
            for iter_step in range(3):
                bar.attach()
                t0 = time.time()
                for i in range(20):
                    bar.monitor_progress()
                    time.sleep(0.2)
                bar.detach() # detach first
                bar.update_lta(time.time() - t0) # then update lta

        """
        self.update_delay = update_delay
        self.old_time = None
        self.lta = None  # last time of arrival (for ETA monitoring)
        self.n_steps = 0  # number of progress bar steps

    def attach(self):
        # setup progress bar
        prefix = "  \033[31mProgress\033[0m:    "
        suffix = "" if self.lta is None else " ETA {0:.3f} s ".format(self.lta)
        sys.stdout.write(prefix + "[=> " + suffix)
        sys.stdout.flush()
        self.old_time = time.time()

    def monitor_progress(self):
        if self.old_time is None:
            self.WrongInitializationError.report()
            raise self.WrongInitializationError

        if time.time() - self.old_time > self.update_delay:
            # update the progress bar every 'self.update_delay' seconds
            if self.lta is None:
                n_back = 2
                suffix = ""
            else:
                eta = self.lta - (self.update_delay * self.n_steps)
                n_back = 2 + len(" ETA {0:.3f} s ".format(eta))
                suffix = " ETA {0:.3f} s ".format(eta)
            sys.stdout.write("\b" * n_back + "=> " + suffix)
            sys.stdout.flush()
            self.old_time = time.time()
            self.n_steps += 1

    def update_lta(self, lta):
        """ Update LTA with the value of the last time of arrival. """
        self.lta = lta

    def detach(self):
        if self.old_time is None:
            self.WrongInitializationError.report()
            raise self.WrongInitializationError

        n_back = 2 if self.lta is None \
            else 2 + len(" ETA {0:.3f} s ".format(self.lta - (self.update_delay * self.n_steps)))

        sys.stdout.write("\b" * n_back + "=]\n")  # this ends the progress bar

        # reset variable:
        self.n_steps = 0

    class WrongInitializationError(Exception):
        """ Raised if the progress bar is not correctly initialized """

        @staticmethod
        def report():
            instr_0 = 'bar = ProgressBar()'
            instr_1 = 'bar.attach()'
            instr_2 = 'bar.monitor_progress()'
            instr_3 = 'bar.detach()'
            print("\033[91m\nRemember that the correct procedure to use a ProgressBar is the following:"
                  "\n{4}{0}\n{4}{1}\n{4}{2}\n{4}{3}\n\033[0m".format(instr_0, instr_1, instr_2, instr_3, ' ' * 3))
            time.sleep(0.2)
