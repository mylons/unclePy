from hdf5 import HDF5
import pandas as pd
import numpy as np


class DLS(HDF5):
    """
    Subclass of HDF5 class, specific for DLS portion of experiment

    Attributes
    ----------
    Reference HDF5 class

    Methods
    -------

    """

    def __init__(self, file_path):
        super().__init__(file_path)
        # Hydrodynamic diameter is consistently 2x values found in .uni file
        # Is this possibly because values are radii and export is diameter?
        self.factor = 2

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR DLS BUNDLE                                          #
    # ----------------------------------------------------------------------- #
    def dls_intensity(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            Intensity hydrodynamic diameter(nm) and amplitude for a single well
        """
        well_num = self.well_name_to_num(well)
        inten = self.file['Application1']['Run1'][well_num]['DLS_Data'] \
            ['DLS0001']['ExperimentAveraged']['AverageCorrelation'] \
            ['Intensity']['Data'][:]
        inten[:, 0] *= self.factor
        df = pd.DataFrame(inten, columns = ['Hydrodynamic Diameter (nm)',
                                            'Amplitude'])
        return df

    def dls_mass(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Mass hydrodynamic diameter(nm) and amplitude for a single well
        """
        well_num = self.well_name_to_num(well)
        mass = self.file['Application1']['Run1'][well_num]['DLS_Data'] \
            ['DLS0001']['ExperimentAveraged']['AverageCorrelation'] \
            ['Mass']['Data'][:]
        mass[:, 0] *= self.factor
        df = pd.DataFrame(mass, columns = ['Hydrodynamic Diameter (nm)',
                                           'Amplitude'])
        return df

    def dls_correlation(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Correlation time(sec) and amplitude for a single well
        """
        well_num = self.well_name_to_num(well)
        corr = self.file['Application1']['Run1'][well_num]['DLS_Data'] \
            ['DLS0001']['ExperimentAveraged']['AverageCorrelation'] \
            ['Correlations'][:]
        # Swap columns to align with typical export
        corr = corr[:, [1, 0]]
        df = pd.DataFrame(corr, columns = ['Time (s)', 'Amplitude'])
        return df

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR DLS SUMMARY                                         #
    # ----------------------------------------------------------------------- #
    def dls_sum_color(self, well):
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

    def dls_sum_temperatures(self):
        """
        Returns
        -------

        """
        wells = self.wells()
        temps = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            temps = np.append(temps,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001']
                              ['ExperimentAveraged']['AverageCorrelation'].
                              attrs['Temperature'])
        return temps

    def dls_sum_zave_diam(self, raw = True):
        """
        Parameters
        ----------
        raw : bool (default = True)
            Return raw or modified values
            Modified values have upper limit of 1000, therefore any value
                greater than 1000 will be returned as ">1000"

        Returns
        -------
        np.array
            If raw values, values are floats
            If not raw values, values are strings (due to ">1000")
        """
        wells = self.wells()
        diams = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            radius = self.file['Application1']['Run1'][well_num] \
                ['DLS_Data']['DLS0001'] \
                ['ExperimentAveraged']['AverageCorrelation'] \
                .attrs['Radius']
            diam = radius * self.factor
            if not raw and diam > 1000:
                diam = '>1000'
            diams = np.append(diams, diam)
        return diams

    def dls_sum_zave_diff_coeff(self):
        """

        Returns
        -------

        """
        pass

    def dls_sum_sd_diam(self, raw = True):
        """
        Parameters
        ----------
        raw : bool (default = True)
            Return raw or modified values
            Modified values have upper limit of 1000, therefore any value
                greater than 1000 will be returned as ">1000"

        Returns
        -------
        np.array
            If raw values, values are floats
            If not raw values, values are strings (due to ">1000")
        """
        wells = self.wells()
        diams = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            radius = self.file['Application1']['Run1'][well_num] \
                ['DLS_Data']['DLS0001'] \
                ['ExperimentAveraged']['AverageCorrelation'] \
                .attrs['StdDev']
            diam = radius * self.factor
            if not raw and diam > 1000:
                diam = '>1000'
            diams = np.append(diams, diam)
        return diams

    def dls_sum_pdi(self):
        """
        PDI = (σ/d)^2
        σ = standard deviation
        d = mean particle diameter

        Returns
        -------
        np.array
            PDI (polydispersity index) values for all wells
        """
        wells = self.wells()
        zave_diams = self.dls_sum_zave_diam(raw = True)
        stdev_diams = self.dls_sum_sd_diam(raw = True)
        pdis = []
        for s, z in zip(stdev_diams, zave_diams):
            pdis = np.append(pdis, ((s / z) ** 2))

        return pdis

    def dls_sum_fit_var(self):
        """

        Returns
        -------

        """
        pass

    def dls_sum_intensity(self):
        """

        Returns
        -------

        """
        wells = self.wells()
        inten = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            inten = np.append(inten,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001']
                              ['ExperimentAveraged']['AverageCorrelation'].
                              attrs['AverageIntensity'])
        return inten

    def dls_sum_pk_mode_diam(self):
        # TODO can you iterate over all PK here?
        """

        Returns
        -------

        """
        pass

    def dls_sum_pk_est_mw(self):
        # TODO can you iterate over all PK here?
        """

        Returns
        -------

        """
        pass

    def dls_sum_pk_poly(self):
        # TODO can you iterate over all PK here?
        """

        Returns
        -------

        """
        pass

    def dls_sum_data_filter(self):
        """

        Returns
        -------

        """
        wells = self.wells()
        dataf = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            dataf = np.append(dataf,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data'].attrs['Data Filter Name'].
                              decode('utf-8'))
        return dataf

    def dls_sum_viscosity(self):
        """

        Returns
        -------

        """
        wells = self.wells()
        visco = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            visco = np.append(visco,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].attrs['Viscosity'])
        return visco

    def dls_sum_ri(self):
        """

        Returns
        -------

        """
        wells = self.wells()
        refin = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            refin = np.append(refin,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].
                              attrs['Refractive Index'])
        return refin

    def dls_sum_der_intensity(self):
        """

        Returns
        -------

        """
        pass

    def dls_sum_min_pk_area(self):
        """

        Returns
        -------

        """
        wells = self.wells()
        minpa = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            minpa = np.append(minpa,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data'].attrs['Minimum Area'])
        return minpa

    def dls_sum_min_rh(self):
        """

        Returns
        -------

        """
        wells = self.wells()
        minrh = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            minrh = np.append(minrh,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data'].attrs['Minimum Rh'])
        return minrh

    # ----------------------------------------------------------------------- #
    # WRITE DATA TO CSV                                                       #
    # ----------------------------------------------------------------------- #
    def write_dls_bundle_csv(self, save_directory):
        """
        Parameters
        ----------
        save_directory : str
            Directory to save CSVs to
            Each well has 3 files: Intensity, Mass, Correlation

        Returns
        -------
        None
        """
        wells = self.wells()
        for well in wells:
            inten_df = self.dls_intensity(well)
            mass_df = self.dls_mass(well)
            corr_df = self.dls_correlation(well)

            inten_df.to_csv(save_directory + '/Intensity-{}-15.csv'.
                            format(well), index = False)
            mass_df.to_csv(save_directory + '/Mass-{}-15.csv'.
                           format(well), index = False)
            corr_df.to_csv(save_directory + '/Correlation-{}-15.csv'.
                           format(well), index = False)


h3 = DLS('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
h4 = DLS('/Users/jmiller/Desktop/UNcle Files/uni files/Gen6 uni 1,2,3.uni')
save_directory = '/Users/jmiller/Desktop/UNcle Files/Misc/'
