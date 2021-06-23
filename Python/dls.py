from hdf5 import HDF5
import pandas as pd
import numpy as np
from scipy.constants import Boltzmann, convert_temperature


class DLS(HDF5):
    """
    Subclass of HDF5 class, specific for DLS portion of experiment

    Attributes
    ----------
    Reference HDF5 class

    Methods
    -------
    dls_intensity(well)
        Returns hydrodynamic diameter and amplitude for single well
    dls_mass(well)
        Returns mass hydrodynamic diameter and amplitude for single well
    dls_correlation(well)
        Returns correlation time and amplitude for single well
    dls_sum_color(well)
        *** NOT CURRENTLY IMPLEMENTED ***
        Returns color for single well
    dls_sum_temperature()
        Returns temperatures used for all wells
    dls_sum_zave_diam(raw, diam)
        Returns Z-average diameters/radii for all wells
    dls_sum_zave_diff_coeff()
        Returns Z-average differential coefficients for all wells
    dls_sum_sd_diam(raw)
        Returns standard deviation of diameter for all wells
    dls_sum_pdi()
        Returns polydispersity index for all wells
    dls_sum_fit_var()
        # TODO need to incorporate
    dls_sum_intensity()
        Returns intensities for all wells
    dls_sum_pk_mode_diam(raw, diam)
        Returns modal diameters/radii for all peaks for all wells
    dls_sum_pk_est_mw(raw)
        Returns estimated molecular weights for all peaks for all wells
    dls_sum_pk_poly()
        Returns polydispersity percentage for all peaks for all wells
    dls_sum_data_filter()
        Returns filter used for all wells
    dls_sum_viscosity()
        Returns viscosity for all wells
    dls_sum_ri()
        Returns refractive index for all wells
    dls_atten_perc()
        Returns attenuation percentage for all wells
    dls_laser_perc()
        Returns laser percentage for all wells
    dls_sum_der_intensity()
        Returns derived intensiteis for all wells
    dls_sum_min_pk_area()
        Returns minimum peak area for all wells
    dls_sum_min_rh()
        Returns minimum relative humidity for all wells
    write_dls_bundle_csv(save_directory)
        Writes 3 files (Intensity, Mass, Correlation) for each well
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
        pd.DataFrame
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
        pd.DataFrame
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
        pd.Series
            unit: °C
            Temperatures used in DLS analysis for all wells
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
        return pd.Series(temps)

    def dls_sum_zave_diam(self, raw = True, diam = True):
        """
        Parameters
        ----------
        raw : bool (default = True)
            Return raw or modified values
            Modified values have upper limit of 1000, therefore any value
                greater than 1000 will be returned as ">1000"
        diam : bool (default = True)
            Return diameters (True) or radii (False)

        Returns
        -------
        pd.Series
            unit: nanometer (nm)
            If raw values, values are floats
            If not raw values, values are strings (due to ">1000")
        """
        wells = self.wells()
        vals = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            val = self.file['Application1']['Run1'][well_num] \
                ['DLS_Data']['DLS0001'] \
                ['ExperimentAveraged']['AverageCorrelation'] \
                .attrs['Radius']
            if diam:
                val = val * self.factor
            if not raw and val > 1000:
                val = '>1000'
            vals = np.append(vals, val)
        return pd.Series(vals)

    def dls_sum_zave_diff_coeff(self):
        """
        Returns
        -------
        pd.Series
            Z-average differential coefficient for all wells
        """
        abs_temp = convert_temperature(self.dls_sum_temperatures(),
                                       'Celsius', 'Kelvin')  # C to K
        rad_m = self.dls_sum_zave_diam(diam = False) / 1000000000  # nm to m
        visco = self.dls_sum_viscosity() / 1000  # cP to kg/m-s
        coef = (Boltzmann * abs_temp) / (6 * np.pi * visco * rad_m)
        return coef

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
        pd.Series
            unit: nanometer (nm)
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
        return pd.Series(diams)

    def dls_sum_pdi(self):
        """
        PDI = (σ/d)^2
        σ = standard deviation
        d = mean particle diameter

        Returns
        -------
        pd.Series
            PDI (polydispersity index) values for all wells
        """
        zave_diams = self.dls_sum_zave_diam(raw = True, diam = True)
        stdev_diams = self.dls_sum_sd_diam(raw = True)
        pdis = []
        for s, z in zip(stdev_diams, zave_diams):
            pdis = np.append(pdis, ((s / z) ** 2))
        return pd.Series(pdis)

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
        pd.Series
            Intensities (cps) for all wells
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
        return pd.Series(inten)

    def dls_sum_pk_mode_diam(self, raw = True, diam = True):
        """
        Parameters
        ----------
        raw : bool (default = True)
            Return raw or modified values
            Modified values have upper limit of 1000, therefore any value
                greater than 1000 will be returned as ">1000"
        diam : bool (default = True)
            Return diameters (True) or radii (False)

        Returns
        -------
        pd.DataFrame
            Peak modal diameters for all peaks for all wells
        """
        wells = self.wells()
        diams = {}
        for well in wells:
            well_num = self.well_name_to_num(well)
            peaks = [i for i in self.file['Application1']['Run1'][well_num]
                     ['DLS_Data']['DLS0001']['ExperimentAveraged']
                     ['AverageCorrelation']['Intensity'].keys()
                     if 'peak' in i.lower()]
            for peak in peaks:
                val = self.file['Application1']['Run1'][well_num] \
                    ['DLS_Data']['DLS0001']['ExperimentAveraged'] \
                    ['AverageCorrelation']['Intensity'][peak] \
                    .attrs['Max'].item()
                diams.setdefault(well, []).append(val)
        df = pd.DataFrame.from_dict(diams, orient = 'index')
        if diam:
            df = 2 * df
        if not raw:
            df[df > 1000] = '>1000'
        return df

    def dls_sum_pk_est_mw(self, raw = True):
        """
        NOTE: What is the highest value before "Out of Range"?
              So far, the highest value seen that still displays: 23,983.70

        Returns
        -------
        pd.DataFrame
            Estimated molecular weights for all peaks for all wells
        """
        rad = self.dls_sum_pk_mode_diam(raw = True, diam = False)
        mw = 2.75 * (rad**2.49)
        if not raw:
            mw[mw > 25000] = 'Out of Range'  # Adjust accordingly to note above
        return mw

    def dls_sum_pk_poly(self):
        """
        peak polydispersity (%) = (peak std / peak mean) * 100

        Returns
        -------
        pd.DataFrame
            Polydispersity percentage for all peaks for all wells
        """
        wells = self.wells()
        pk_poly = {}
        for well in wells:
            well_num = self.well_name_to_num(well)
            path = self.file['Application1']['Run1'][well_num] \
                ['DLS_Data']['DLS0001']['ExperimentAveraged'] \
                ['AverageCorrelation']['Intensity']
            peaks = [i for i in path.keys() if 'peak' in i.lower()]
            for peak in peaks:
                std = path[peak].attrs['Std'].item()
                mean = path[peak].attrs['Mean'].item()
                pk_poly.setdefault(well, []).append(100 * std / mean)
        df = pd.DataFrame.from_dict(pk_poly, orient = 'index')
        return df

    def dls_sum_data_filter(self):
        """
        Returns
        -------
        pd.Series
            Filter used in DLS analysis for all wells
        """
        wells = self.wells()
        dataf = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            dataf = np.append(dataf,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data'].attrs['Data Filter Name'].
                              decode('utf-8'))
        return pd.Series(dataf)

    def dls_sum_viscosity(self):
        """
        Returns
        -------
        pd.Series
            unit: centipoise (cP)
            Viscosity found in DLS analysis for all wells
        """
        wells = self.wells()
        visco = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            visco = np.append(visco,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].attrs['Viscosity'])
        return pd.Series(visco)

    def dls_sum_ri(self):
        """
        Returns
        -------
        pd.Series
            Refractive index used in DLS analysis for all wells
        """
        wells = self.wells()
        refin = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            refin = np.append(refin,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].
                              attrs['Refractive Index'])
        return pd.Series(refin)

    def dls_atten_perc(self):
        """
        Returns
        -------
        pd.Series
            Attenuation percentage used for all wells.
            Used for calculating derived intensity
            Returned as float percentage, i.e. 75% = 75.0
        """
        wells = self.wells()
        atten = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            atten = np.append(atten,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].
                              attrs['Attenuation %'])
        return pd.Series(atten)

    def dls_laser_perc(self):
        """
        Returns
        -------
        pd.Series
            Laser percentage used for all wells.
            Used for calculating derived intensity
            Returned as float percentage, i.e. 75% = 75.0
        """
        wells = self.wells()
        laser = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            laser = np.append(laser,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].
                              attrs['Laser %'])
        return pd.Series(laser)

    def dls_sum_der_intensity(self):
        """
        Derived intensity = intensity / attenuation % / laser %

        Returns
        -------
        pd.Series
            unit: counts per second (cps)
            Derived intensities for all wells
        """
        inten = self.dls_sum_intensity()
        atten = self.dls_atten_perc() / 100
        laser = self.dls_laser_perc() / 100
        dis = []
        for i, a, l in zip(inten, atten, laser):
            dis = np.append(dis, (i / a / l))
        return pd.Series(dis)

    def dls_sum_min_pk_area(self):
        """
        Returns
        -------
        pd.Series
            Minimum peak area used in DLS analysis for all wells
            Returned as float percentage, i.e. 75% = 75.0
        """
        wells = self.wells()
        minpa = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            minpa = np.append(minpa,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data'].attrs['Minimum Area'])
        return pd.Series(minpa)

    def dls_sum_min_rh(self):
        """
        Returns
        -------
        pd.Series
            Minimum relative humidity used in DLS analysis for all wells
        """
        wells = self.wells()
        minrh = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            minrh = np.append(minrh,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data'].attrs['Minimum Rh'])
        return pd.Series(minrh)

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
