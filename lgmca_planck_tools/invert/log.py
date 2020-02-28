from __future__ import division, print_function

import asyncio
import logging
import os
import sys
import time


null_logger = logging.Logger('null_logger')
null_logger.addHandler(logging.NullHandler())


def make_logger(name,
                log_file=os.path.join(os.getcwd(), 'log.txt'),
                toplevel_log_file=os.path.join(os.environ['HOME'], 'lgmca_postprocessing_log.txt'),
                formatter='[%(asctime)s] [%(name)s] (%(process)s) %(levelname)s: %(message)s',
                debug=True):
    '''
    Creates a Logger for use throughout... this whole thing.
    '''
    formatter = logging.Formatter(formatter)

    toplevel = logging.getLogger('lgmca_processing')
    if not toplevel.handlers:
        toplevel.setLevel(logging.INFO)
        toplevel.propagate = False
        if toplevel_log_file:
            tplvl_hndlr = logging.FileHandler(toplevel_log_file)
            tplvl_hndlr.setFormatter(formatter)
            tplvl_hndlr.setLevel(logging.INFO)
            toplevel.addHandler(tplvl_hndlr)

    logger = logging.getLogger('lgmca_processing.{}'.format(name))
    assert logger.parent == toplevel
    logger.setLevel(logging.DEBUG)

    # Two separate handlers: to stdout, and to a log file
    # Only print INFO & up to console, but print everything to log file
    strm_hndlr = logging.StreamHandler(sys.stdout)
    strm_hndlr.setLevel(logging.INFO)
    strm_hndlr.setFormatter(formatter)
    logger.addHandler(strm_hndlr)

    if log_file:
        dbg_hndlr = logging.FileHandler(log_file)
        if debug:
            dbg_hndlr.setLevel(logging.DEBUG)
        else:
            dbg_hndlr.setLevel(logging.INFO)
        dbg_hndlr.setFormatter(formatter)
        logger.addHandler(dbg_hndlr)

    return logger


async def run_command(cmd, timeout=1e-3, limit=1000, logger=null_logger):
    '''
    Runs a shell command (given by `cmd`, a list of strings of the command and
    its arguments) and logs its stdout and stderr to debug and error,
    respectively.

    Returns the process's return code.
    '''
    loop = asyncio.get_running_loop()
    def generator():
        return asyncio.subprocess.SubprocessStreamProtocol(limit, loop)

    transport, protocol = await loop.subprocess_exec(generator, *cmd, stdin=None)

    while not (protocol.stdout.at_eof() and protocol.stderr.at_eof()):
        stdout, stderr = [asyncio.ensure_future(f.readline())
                          for f in (protocol.stdout, protocol.stderr)]
        done, pending = await asyncio.wait((stdout, stderr), timeout=timeout)

        # For whichever readline()s gave results, record the results
        for obj in done:
            if obj == stdout:
                level = logger.debug
            elif obj == stderr:
                level = logger.error
            bline = await obj
            line = str(bline, 'utf-8')
            if line:
                level(line.strip())

        # Cancel whichever readline()s didn't work
        for obj in pending:
            obj.cancel()

    rc = transport.get_returncode()
    if rc is None:
        raise ValueError('rc should not be None')
    if rc != 0:
        logger.error('{} returned exit error code: {}'.format(cmd[0], rc))
    else:
        logger.info('{} returned exit code: {}'.format(cmd[0], rc))

    return rc


class Timer(object):
    '''
    Times a chunk of code and records in the log.
    '''
    def __init__(self, logger, msg, *args, **kwargs):
        self._logger = logger
        self._msg = msg
        self._args = args
        self._kwargs = kwargs
        self.level = logging.DEBUG

    def with_level(self, new_level):
        '''
        Change the logging level of this Timer. The default is DEBUG.
        '''
        self.level = new_level
        return self

    def __enter__(self):
        self._start = time.time()
        self._logger.log(level=self.level,
                         msg='(starting) ' + self._msg.format(*self._args, **self._kwargs))

    def __exit__(self, exc_type, exc_val, exc_tb):
        diff = time.time() - self._start
        del self._start

        # Handle error & non-error differently
        if exc_type is None:
            self._logger.log(level=self.level,
                             msg='(finished) ' + self._msg.format(*self._args, **self._kwargs))
            self._logger.log(level=self.level,
                             msg='(finished) time taken: {:0.3f} sec'.format(diff))
        else:
            # Record as error
            self._logger.log(level=logging.ERROR,
                             msg='(error)    ' + self._msg.format(*self._args, **self._kwargs),
                             exc_info=(exc_type, exc_val, exc_tb))
            self._logger.log(level=logging.ERROR,
                             msg='(error)    time taken: {:0.3f} sec'.format(diff))
