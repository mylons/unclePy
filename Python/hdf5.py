import h5py
import numpy as np


class HDF5:
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')

    # ----------------------------------------------------------------------- #
    # BASIC DATA COLLECTION                                                   #
    # ----------------------------------------------------------------------- #
    def wells(self):
        """
        Returns
        -------
        np.array
            Well names

        Examples
        -------
        np.array(['A1', 'B1', ...])
        """
        wells = []
        for i in self.file['Application1']['Run1']['SampleData']:
            wells = np.append(wells, i[0].decode('utf-8'))
        return wells

    def well_name_to_num(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        string
            Well number, e.g. 'Well_01'
        """
        well_num = np.argwhere(self.wells() == well)[0][0] + 1
        well_num = f'Well_{well_num:02}'
        return well_num

    def samples(self):
        """
        Returns
        -------
        np.array
            Sample names

        Examples
        -------
        np.array(['0.1 mg/ml Uni A1', '0.1 mg/ml Uni B1', ...])
        """
        samples = []
        for i in self.file['Application1']['Run1']['SampleData']:
            samples = np.append(samples, i[1].decode('utf-8'))
        return samples


def verify(value):
    """
    Parameters
    ----------
    value : int, float
        Any value to verify is legitimate

    Returns
    -------
    int, float, np.nan
        Depends on input value
        Returns input value if valid, otherwise np.nan

    """
    if value != -1:
        return value
    else:
        return np.nan


h1 = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
h2 = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/Gen6 uni 1,2,3.uni')
save_path = '/Users/jmiller/Desktop/UNcle Files/Misc/uncle_out.xlsx'
