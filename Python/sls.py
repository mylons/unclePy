from hdf5 import HDF5, verify
import pandas as pd
import numpy as np


class SLS(HDF5):
    """
    Subclass of HDF5 class, specific for SLS portion of experiment

    Attributes
    ----------
    Reference HDF5 class

    Methods
    -------
    sls_temperature(well)
        Returns temperature used for single well

    sls_spec_times(well)
        Returns times used for single well

    sls_spec_wavelengths(well)
        Returns wavelengths used for single well

    sls_spec_intensity(well, temp)
        Returns intensities found at a single temperature for single well

    sls_summary_color()
        *** Not currently implemented ***

    sls_summary_tms(well)
        Returns all TM values for single well

    sls_summary_tonset(well)
        Returns T_onset value for single well

    sls_summary_tagg266(well)
        Returns T_agg 266 for single well

    sls_summary_tagg473(well)
        Returns T_agg 473 for single well

    sls_bcm(well)
        Returns BCM/nm for single well

    sls_266(well)
        Returns SLS 266 nm/Count for single well with temperature

    sls_473(well)
        Returns SLS 473 nm/Count for single well with temperature

    sls_spec_well(well)
        Returns pd.DataFrame of intensities for single well

    sls_summary()
        Returns pd.DataFrame of summary for entire experiment

    sls_export(well)
        Returns pd.DataFrame of BCM/nm, SLS 266 nm/Count, SLS 473 nm/Count
        for single well

    write_sls_summary_sql(username, password, host, database)
        Saves summary data to PostgreSQL database
    """

    def __init__(self, file_path, uncle_experiment_id, well_set_id):
        super().__init__(file_path, uncle_experiment_id, well_set_id)

    # ----------------------------------------------------------------------- #
    # GENERIC DATA COLLECTION                                                 #
    # ----------------------------------------------------------------------- #
    def sls_temperature(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Temperature used in SLS analysis for single well
        """
        well_num = self.well_name_to_num(well)
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
        well_num = self.well_name_to_num(well)
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
    def sls_spec_wavelengths(self, well):
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
        well_num = self.well_name_to_num(well)
        wavelengths = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['CorrectedSpectra']['0001'][:, 0]
        return wavelengths

    def sls_spec_intensity(self, well, temp):
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
        well_num = self.well_name_to_num(well)
        temps = self.sls_temperature(well)
        # Cannot index into HDF5 dataset, therefore need to call with dict key
        index = np.flatnonzero(temps == temp)[0]
        inten_cnt = f'{index + 1:04}'

        inten_meas = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['CorrectedSpectra'][inten_cnt]

        # Assert we are looking at correct well, temp, etc.
        np.testing.assert_array_equal(inten_meas[:, 0],
                                      self.sls_spec_wavelengths(well))

        return inten_meas[:, 1]

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR SLS SUMMARY                                         #
    # ----------------------------------------------------------------------- #
    @staticmethod
    def sls_summary_color():
        """
        NOTE: Datasets have not included this yet, therefore unable to locate
              where it is captured in .uni file.

        Returns
        -------
        np.nan
        """
        return np.nan

    def sls_summary_tms(self, well):
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
        --------
            np.array([42.48, 82.4 ,  0.  ,  0.  ])
        """
        well_num = self.well_name_to_num(well)
        tms = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['Tms'][0]
        return tms

    def sls_summary_tonset(self, well):
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
        well_num = self.well_name_to_num(well)
        tonset = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['TonsetBCM'][0]
        tonset = verify(tonset)
        return tonset

    def sls_summary_tagg266(self, well):
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
        well_num = self.well_name_to_num(well)
        tagg266 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['Tagg266'][0]
        tagg266 = verify(tagg266)
        return tagg266

    def sls_summary_tagg473(self, well):
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
        well_num = self.well_name_to_num(well)
        tagg473 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['Tagg473'][0]
        tagg473 = verify(tagg473)
        return tagg473

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR SLS BCM, 266, 473                                   #
    # ----------------------------------------------------------------------- #
    def sls_bcm(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            BCM/nm for single well with temperature
        """
        well_num = self.well_name_to_num(well)
        temperature = self.sls_temperature(well)
        sls_bcm = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['BCM'][:]
        sls_bcm_data = {'uncle_summary_id': self.well_name_to_summary(well),
                        'temperature': temperature,
                        'bcm': sls_bcm}
        df = pd.DataFrame(sls_bcm_data,
                          columns = ['uncle_summary_id',
                                     'temperature',
                                     'bcm'])
        return df

    def sls_266(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            SLS 266 nm/Count for single well with temperature
        """
        well_num = self.well_name_to_num(well)
        temperature = self.sls_temperature(well)
        sls_266 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['SLS266'][:]
        sls_266_data = {'uncle_summary_id': self.well_name_to_summary(well),
                        'temperature': temperature,
                        'sls_266': sls_266}
        df = pd.DataFrame(sls_266_data,
                          columns = ['uncle_summary_id',
                                     'temperature',
                                     'sls_266'])
        return df

    def sls_473(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            SLS 266 nm/Count for single well with temperature
        """
        well_num = self.well_name_to_num(well)
        temperature = self.sls_temperature(well)
        sls_473 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['SLS473'][:]
        sls_473_data = {'uncle_summary_id': self.well_name_to_summary(well),
                        'temperature': temperature,
                        'sls_473': sls_473}
        df = pd.DataFrame(sls_473_data,
                          columns = ['uncle_summary_id',
                                     'temperature',
                                     'sls_473'])
        return df

    # ----------------------------------------------------------------------- #
    # DATAFRAME ASSEMBLY                                                      #
    # ----------------------------------------------------------------------- #
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
        temps = self.sls_temperature(well)
        times = self.sls_times(well)
        waves = self.sls_spec_wavelengths(well)
        inten = [pd.Series(self.sls_spec_intensity(well, temp))
                 for temp in temps]

        # NOTE: inconsistent whitespace here is purposeful to match file
        cols = [f'Temp :{temps[i]:02f}, Time:{times[i]:.1f}'
                for i in range(len(temps))]

        # Transpose due to the way dataframe is compiled
        df = pd.DataFrame(data = inten).T
        df.columns = cols
        df.index = waves
        df.index.name = 'Wavelength'

        return df

    def sls_summary(self):
        """
        Returns
        -------
        pd.DataFrame
            Full dataframe for SLS Summary
        """
        wells = self.wells()
        samples = self.samples()

        cols = ['color', 'well_id', 'sample', 't_onset', 't_agg_266',
                't_agg_473']
        # Determine number of Tm columns
        tm_cols = []
        for well in wells:
            tm_cols = np.append(tm_cols,
                                np.flatnonzero(self.sls_summary_tms(well)))
        max_tm = int(np.max(tm_cols) + 1)
        cols.extend(['t_m_{}'.format(i + 1) for i in range(max_tm)])

        df = pd.DataFrame(columns = cols)

        for well in wells:
            well_summary = {'color': self.sls_summary_color(),
                            'well_id': self.well_name_to_id(well),
                            't_onset': self.sls_summary_tonset(well),
                            't_agg_266': self.sls_summary_tagg266(well),
                            't_agg_473': self.sls_summary_tagg473(well)}

            sample_index = np.flatnonzero(
                np.char.find(self.samples(), well) != -1)
            well_summary['sample'] = samples[sample_index][0]

            tms = self.sls_summary_tms(well)
            for i in range(max_tm):
                well_summary['t_m_{}'.format(i + 1)] = \
                    tms[i] if tms[i] != 0 else np.nan

            df = df.append(well_summary, ignore_index = True)

        return df

    # ----------------------------------------------------------------------- #
    # WRITE DATA TO POSTGRESQL                                                #
    # ----------------------------------------------------------------------- #
    def write_sls_266_sql(self):
        """
        Returns
        -------
        None
        """
        self.exp_confirm_created()

        wells = self.wells()
        df = pd.DataFrame(columns =
                          ['uncle_summary_id',
                           'temperature',
                           'sls_266'])
        for well in wells:
            df_266 = self.sls_266(well)
            df = df.append(df_266).reset_index(drop = True)
        df.name = 'sls_266'
        df = self.df_to_sql(df)
        df.to_sql('uncle_sls_266',
                  self.engine,
                  if_exists = 'append',
                  index = False)

    def write_sls_473_sql(self):
        """
        Returns
        -------
        None
        """
        self.exp_confirm_created()

        wells = self.wells()
        df = pd.DataFrame(columns =
                          ['uncle_summary_id',
                           'temperature',
                           'sls_473'])
        for well in wells:
            df_473 = self.sls_473(well)
            df = df.append(df_473).reset_index(drop = True)
        df.name = 'sls_473'
        df = self.df_to_sql(df)
        df.to_sql('uncle_sls_473',
                  self.engine,
                  if_exists = 'append',
                  index = False)

    def write_sls_bcm_sql(self):
        """
        Returns
        -------
        None
        """
        self.exp_confirm_created()

        wells = self.wells()
        df = pd.DataFrame(columns =
                          ['uncle_summary_id',
                           'temperature',
                           'bcm'])
        for well in wells:
            df_bcm = self.sls_bcm(well)
            df = df.append(df_bcm).reset_index(drop = True)
        df.name = 'sls_bcm'
        df = self.df_to_sql(df)
        df.to_sql('uncle_sls_bcm',
                  self.engine,
                  if_exists = 'append',
                  index = False)


"""
h5 = SLS('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
h6 = SLS('/Users/jmiller/Desktop/UNcle Files/uni files/Gen6 uni 1,2,3.uni')
save_path = '/Users/jmiller/Desktop/UNcle Files/Misc/'
"""
