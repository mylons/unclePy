import h5py
import pandas as pd
import numpy as np
import string


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

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR SLS SPEC                                            #
    # ----------------------------------------------------------------------- #
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
                          ['Fluor_SLS_Data']['CorrectedSpectra']['0001'][:, 0]
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

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR SLS SUMMARY                                         #
    # ----------------------------------------------------------------------- #
    def sls_color(self, well):
        """
        TODO: color is currently blank for all files. Is there ever a value?

        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        None (because currently not used)
        """
        return np.nan

    def sls_tms(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            All TM values for single well

        Examples
        -------
            np.array([42.48, 82.4 ,  0.  ,  0.  ])
        """
        well_num = well_name_to_num(well)
        tms = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['Tms'][0]
        return tms

    def sls_tonset(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        float
            Tonset value for single well
        """
        well_num = well_name_to_num(well)
        tonset = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['TonsetBCM'][0]
        return tonset

    def sls_tagg266(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        float
            Tagg266 for single well
        """
        well_num = well_name_to_num(well)
        tagg266 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['Tagg266'][0]
        return tagg266

    def sls_tagg473(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        float
            Tagg473 for single well
        """
        well_num = well_name_to_num(well)
        tagg473 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['Tagg473'][0]
        return tagg473

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
        df.index.name = 'Wavelength'

        return df

    def sls_sum(self):
        """
        Returns
        -------
        pd.DataFrame
            Full dataframe for SLS Summary
        """
        wells = self.wells()
        samples = self.samples()

        cols = ['Color', 'Well', 'Sample', 'Tonset (°C)', 'Tagg 266 (°C)',
                'Tagg 473 (°C)']
        # Determine how many Tm columns
        tm_cols = []
        for i in self.wells():
            tm_cols = np.append(tm_cols, np.flatnonzero(self.sls_tms(i)))
        max_tm = int(np.max(tm_cols) + 1)
        cols.extend(['Tm{} (°C)'.format(i + 1) for i in
                     range(max_tm)])

        df = pd.DataFrame(columns = cols)

        for i in wells:
            well_sum = {'Color': self.sls_color(i),
                        'Well': i,
                        'Tonset (°C)': self.sls_tonset(i),
                        'Tagg 266 (°C)': self.sls_tagg266(i),
                        'Tagg 473 (°C)': self.sls_tagg473(i)}

            sample_index = np.flatnonzero(
                np.char.find(self.samples(), i) != -1)
            well_sum['Sample'] = samples[sample_index][0]

            tms = self.sls_tms(i)
            for j in range(max_tm):
                well_sum['Tm{} (°C)'.format(j + 1)] = \
                    tms[j] if tms[j] != 0 else np.nan

            df = df.append(well_sum, ignore_index = True)

        return df

    def write_sls_spec_excel(self, save_path):
        """
        Parameters
        ----------
        save_path
            Directory to save Excel file to

        Returns
        -------
        None
        """
        wells = self.wells()
        with pd.ExcelWriter(save_path) as writer:
            for well in wells:
                df = self.sls_spec_well(well)
                df.to_excel(writer, sheet_name = well)

    def write_sls_sum_excel(self, save_path):
        """
        Parameters
        ----------
        save_path
            Directory to save Excel file to

        Returns
        -------
        None
        """
        df = self.sls_sum()
        df.to_excel(save_path, index = False)


def well_name_to_num(well):
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
    well_num = string.ascii_uppercase.index(well[0]) + 1
    well_num = f'Well_{well_num:02}'
    return well_num


h = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
save_path = '/Users/jmiller/Desktop/UNcle Files/Misc/uncle_out.xlsx'
