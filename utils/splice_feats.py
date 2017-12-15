#!/usr/bin/env python3

import sys

if len(sys.argv) != 2:
    print("Must specify feature directory")
    sys.exit(1)

feature_dir = sys.argv[1]

from multiprocessing import Pool
import os
import subprocess as sp

def spliceScp(scp_filename):
    ark_spliced_filename = "%s_spliced.ark" % scp_filename.rstrip(".scp")
    scp_spliced_filename = "%s_spliced.scp" % scp_filename.rstrip(".scp")
    splice_args = ["splice-feats",
                   "--left-context=%d" % int(os.environ["LEFT_SPLICE"]),
                   "--right-context=%d" % int(os.environ["RIGHT_SPLICE"]),
                   "scp:%s" % scp_filename,
                   "ark,scp:%s,%s" % (ark_spliced_filename, scp_spliced_filename)]
    completed_process = sp.run(splice_args)
    if completed_process.returncode:
        print("Feature splice process failed with code %d" % completed_process.returncode)

scp_filenames = list(filter(lambda x: "_spliced" not in x and "_lstm" not in x,
                            filter(lambda x: "fbank" in x,
                                   filter(lambda x: x.endswith(".scp"), os.listdir(feature_dir)))))
max_threads = 4
num_threads = min(max_threads, len(scp_filenames))
p = Pool(processes=num_threads)

# Fix for multiprocessing exit bug
# https://stackoverflow.com/questions/30786853/exit-from-multiprocessing-pool-upon-exception-or-keyboardinterrupt
try:
    scp_paths = list(map(lambda x: os.path.join(feature_dir, x), scp_filenames))
    result = p.map_async(spliceScp, scp_paths)
    p.close()
    result.wait(timeout=9999999) # Without a timeout, you can't interrupt this.
except KeyboardInterrupt:
    p.terminate()
finally:
    p.join()
