import fcntl
import os
import pathlib
import time

def lock_acquire(lockname):

    lockfile = None
    while lockfile is None:
        try:
            lockfile = lock_acquire_inner(lockname)
        except BlockingIOError:
            time.sleep(0.1)

    return lockfile


def lock_acquire_inner(lockname: str, operation: int = fcntl.LOCK_EX | fcntl.LOCK_NB):
    """
    Acquire the flock lockfile

    Parameters
    ----------
    operation : int, optional
        The flock operation to perform (default is LOCK_EX | LOCK_NB)

    Raises
    ------
    BlockingIOError
        If perform is ORed with LOCK_NB, it raises if the file is
        already acquired by another process.

    Returns
    ------
    File
        The flock lockfile
    """
    #lockname = __file__.split("/")[-1].strip(".py")
    lockfile = open(pathlib.Path().joinpath(f"/tmp/{lockname}.flock"), "w")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)

    return lockfile

def lock_release(lockfile):
    """
    Release the flock lockfile

    Parameters
    ----------
    lockfile: file descriptor, required
        The lock file to release
    """
    fcntl.flock(lockfile, fcntl.LOCK_UN)
    os.close(lockfile.fileno())
