import subprocess

def shell(command, printOut=True):
    """
    Run shell commands in Linux, decide if printing or not the output in console

    Parameters
    ----------
    command: text command
    printOut: decide if print output or not

    Returns
    -------
    None

    """
    if printOut == False:
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=None)
    else:
        proc = subprocess.Popen(command, shell=True)
        proc.wait(timeout=None)