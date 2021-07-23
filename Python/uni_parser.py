from hdf5 import HDF5
from sls import SLS
from dls import DLS
import argparse

parser = argparse.ArgumentParser(description = 'Parse .uni file')
parser.add_argument('uni_file',
                    help = 'Path to .uni file',
                    type = str)
parser.add_argument('uncle_experiment_id',
                    help = 'Database ID for UNcle experiment',
                    type = int)
parser.add_argument('well_set_id',
                    help = 'Database ID for associated well set',
                    type = int)

args = parser.parse_args()

if __name__ == '__main__':
    sls = SLS(args.uni_file, args.uncle_experiment_id)
    dls = DLS(args.uni_file, args.uncle_experiment_id)

    dls.write_dls_bundle_sql('postgres', '', 'localhost', 'ebase_dev')
    dls.write_dls_sum_sql('postgres', '', 'localhost', 'ebase_dev')
    sls.write_sls_bundle_sql('postgres', '', 'localhost', 'ebase_dev')
    sls.write_sls_sum_sql('postgres', '', 'localhost', 'ebase_dev')
