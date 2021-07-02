from hdf5 import HDF5
from sls import SLS
from dls import DLS
import argparse

parser = argparse.ArgumentParser(description = 'Parse .uni file')
parser.add_argument('uni_file',
                    metavar = 'File Path',
                    help = 'Path to .uni file',
                    type = str)

args = parser.parse_args()

if __name__ == '__main__':
    sls = SLS(args.uni_file)
    dls = DLS(args.uni_file)

    dls.write_dls_bundle_sql('postgres', '', 'localhost', 'ebase_dev')
    dls.write_dls_sum_sql('postgres', '', 'localhost', 'ebase_dev')
    sls.write_sls_bundle_sql('postgres', '', 'localhost', 'ebase_dev')
    sls.write_sls_sum_sql('postgres', '', 'localhost', 'ebase_dev')