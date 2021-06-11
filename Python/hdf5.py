import h5py
import pandas as pd
import numpy as np
import string


class HDF5:
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')

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

    def sls_temperatures(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Temperatures used in SLS analysis for single well
        """
        well_num = well_name_to_num(well)
        meas_dir = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['CorrectedSpectra']

        temps = []
        for i in meas_dir:
            temps = np.append(temps,
                              meas_dir[i].attrs['Actual temperature'])
        return temps

    def sls_times(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Times used in SLS analysis for single well
        """
        well_num = well_name_to_num(well)
        meas_dir = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['CorrectedSpectra']

        times = []
        for i in meas_dir:
            times = np.append(times,
                              meas_dir[i].attrs['Actual Time'])
        return times

    def sls_wavelengths(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Wavelengths used in SLS analysis for single well
        """
        well_num = well_name_to_num(well)
        wavelengths = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['CorrectedSpectra'] \
            ['0001'][:, 0]
        return wavelengths

    def sls_intensity(self, well, temp):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'
        temp : float
            Single temperature value from file
            Does a direct lookup so value should be taken directly from file

        Returns
        -------
        np.array
            Intensities for a single temp of single well
        """
        well_num = well_name_to_num(well)
        temps = self.sls_temperatures(well)
        # Cannot index into HDF5 dataset, therefore need to call with dict key
        index = np.flatnonzero(temps == temp)[0]
        inten_cnt = f'{index + 1:04}'

        inten_meas = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['CorrectedSpectra'][inten_cnt]

        # Assert we are looking at correct well, temp, etc.
        np.testing.assert_array_equal(inten_meas[:, 0],
                                      self.sls_wavelengths(well))

        return inten_meas[:, 1]

    def sls_spec_well(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            Full dataframe of SLS intensities for single well
            This is comparable to an Excel tab for one well, e.g. 'A1'

        """
        temps = self.sls_temperatures(well)
        times = self.sls_times(well)
        waves = self.sls_wavelengths(well)
        inten = [pd.Series(self.sls_intensity(well, temp)) for temp in temps]

        cols = [f'Temp :{temps[i]:02f}, Time:{times[i]:.1f}'
                for i in range(len(temps))]

        # Transpose due to the way dataframe is compiled
        df = pd.DataFrame(inten).T
        df.columns = cols
        df.index = waves

        return df


def well_name_to_num(well):
    """
    Parameters
    ----------
    well : str
        Single well name, e.g. 'A1'

    Returns
    -------
    string
        Well number
    """
    well_num = string.ascii_uppercase.index(well[0]) + 1
    well_num = f'Well_{well_num:02}'
    return well_num


h = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
