import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

path_to_data = '/Users/adityabhandari/Desktop/Jobs/INTERVIEWS/UPTAKE - Data Science Intern/Data Science Case Study V2 (amcelhinney)/'
prediction_cols = ['responded', 'amount']
in_test_not_train_col = ['market']


def get_zip_code_cost_df():
    data = []
    with open(path_to_data + 'zipCodeMarketingCosts.csv', 'r') as read_file:
        for line in read_file:
            data += [line.replace('\n', '').replace('"', '').split(',')]

    df = pd.DataFrame(data=data[1:], columns=data[0])
    df.iloc[:, 0].apply(np.float)
    df.iloc[:, 1].apply(np.int)

    return df


def load_data(file_name='test.csv'):
    data = []
    with open(path_to_data + '/data/' + file_name, 'r') as read_file:

        for line in read_file:
            data += [line.replace('\n', '').replace('"', '').split(',')]

    df = pd.DataFrame(data=data[1:], columns=data[0])

    return df


df_zip = get_zip_code_cost_df()
df_test = load_data('test.csv')
df = load_data('train.csv')

# Drop useless columns

drop_cols = ['adate_2', 'adate_3', 'adate_4', 'adate_5',
             'adate_6', 'adate_7', 'adate_8', 'adate_9',
             'adate_10', 'adate_11', 'adate_12', 'adate_13',
             'adate_14', 'adate_15', 'adate_16', 'adate_17',
             'adate_18', 'adate_19', 'adate_20', 'adate_21',
             'adate_22', 'adate_23', 'adate_24',
             'anc1', 'anc2', 'anc3', 'anc4', 'anc5',
             'anc6', 'anc7', 'anc8', 'anc9', 'anc10',
             'anc11', 'anc12', 'anc13', 'anc14', 'anc15',
             'chil1', 'chil2', 'chil3',
             'chilc1', 'chilc2', 'chilc3', 'chilc4', 'chilc5',
             'ethc1', 'ethc2', 'ethc3', 'ethc4', 'ethc5', 'ethc6',
             'hc1', 'hc2', 'hc3', 'hc4', 'hc5', 'hc6', 'hc7',
             'hc8', 'hc9', 'hc10', 'hc11', 'hc12', 'hc13', 'hc14',
             'hc15', 'hc16', 'hc17', 'hc18', 'hc19', 'hc20', 'hc21',
             'hphone_d', 'dob', 'noexch', 'ageflag', 'geocode',
             'lifesrc', 'mdmaud', 'oedc1', 'oedc2', 'oedc3', 'oedc4', 'oedc5',
             'oedc6', 'oedc7', 'ec1', 'ec2', 'ec3', 'ec4', 'ec5', 'ec6', 'ec7',
             'ec8', 'rfa_2', 'rfa_3', 'rfa_4', 'rfa_5', 'rfa_6', 'rfa_7', 'rfa_8',
             'rfa_9', 'rfa_10', 'rfa_11', 'rfa_12', 'rfa_13', 'rfa_14',
             'rfa_15', 'rfa_16', 'rfa_17', 'rfa_18', 'rfa_19', 'rfa_20',
             'rfa_21', 'rfa_22', 'rfa_23', 'rfa_24', 'rfa_2r', 'rfa_2f', 'rfa_2a',
             'mdmaud', 'mdmaud_r', 'mdmaud_f', 'mdmaud_a',
             'rdate_3', 'rdate_4', 'rdate_5', 'rdate_6', 'rdate_7',
             'rdate_8', 'rdate_9', 'rdate_10', 'rdate_11', 'rdate_12',
             'rdate_13', 'rdate_14', 'rdate_15', 'rdate_16', 'rdate_17',
             'rdate_18', 'rdate_19', 'rdate_20', 'rdate_21', 'rdate_22',
             'rdate_23', 'rdate_24',
             'ramnt_3', 'ramnt_4', 'ramnt_5', 'ramnt_6', 'ramnt_7',
             'ramnt_8', 'ramnt_9', 'ramnt_10', 'ramnt_11', 'ramnt_12',
             'ramnt_13', 'ramnt_14', 'ramnt_15', 'ramnt_16', 'ramnt_17',
             'ramnt_18', 'ramnt_19', 'ramnt_20', 'ramnt_21', 'ramnt_22',
             'ramnt_23', 'ramnt_24', 'timelag', 'age901', 'age902', 'age903',
             'age904', 'age905', 'age906', 'age907', 'ramntall', 'has_chapter',
             'cluster', 'cluster2', 'geocode2', 'wealth2', 'lsc1', 'lsc2',
             'lsc3', 'lsc4', 'date', 'source', 'datasrce', 'has_chapter', 'id']


def drop_useless_cols(df):
    return df.drop(drop_cols, axis=1)


df = drop_useless_cols(df)
df_test = drop_useless_cols(df_test)

def weighted_sample(weights, sample_size):
    """
    Returns a weighted sample without replacement. 
    """
    totals = np.cumsum(weights)
    sample = []
    for i in xrange(sample_size):
        rnd = random.random() * totals[-1]
        idx = np.searchsorted(totals,rnd,'right')
        sample.append(idx)
        totals[idx:] -= weights[idx]
    return sample

def transformations(df):

    df.child03 = df.child03.apply(lambda x: 1 if x != ' ' else 0)
    df.child07 = df.child07.apply(lambda x: 1 if x != ' ' else 0)
    df.child12 = df.child12.apply(lambda x: 1 if x != ' ' else 0)
    df.child18 = df.child18.apply(lambda x: 1 if x != ' ' else 0)

    df.homeownr = df.homeownr.apply(lambda x: 1 if x == 'H' else(np.nan if x == 'U' else 0))

    df.mailcode = df.mailcode.apply(lambda x: 0 if x == 'B' else 1)

    df.recinhse = df.recinhse.apply(lambda x: 1 if x == 'X' else 0)
    df.recp3 = df.recp3.apply(lambda x: 1 if x == 'X' else 0)
    df.recpgvg = df.recpgvg.apply(lambda x: 1 if x == 'X' else 0)
    df.recsweep = df.recsweep.apply(lambda x: 1 if x == 'X' else 0)
    df.major = df.major.apply(lambda x: 1 if x == 'X' else 0)

    df.kidstuff = df.kidstuff.apply(lambda x: 1 if x == 'Y' else 0)
    df.collect1 = df.collect1.apply(lambda x: 1 if x == 'Y' else 0)
    df.plates = df.plates.apply(lambda x: 1 if x == 'Y' else 0)
    df.cards = df.cards.apply(lambda x: 1 if x == 'Y' else 0)

    df.bible = df.bible.apply(lambda x: 1 if x == 'Y' else 0)
    df.boats = df.boats.apply(lambda x: 1 if x == 'Y' else 0)
    df.catlg = df.catlg.apply(lambda x: 1 if x == 'Y' else 0)
    df.cdplay = df.cdplay.apply(lambda x: 1 if x == 'Y' else 0)
    df.crafts = df.crafts.apply(lambda x: 1 if x == 'Y' else 0)
    df.fisher = df.fisher.apply(lambda x: 1 if x == 'Y' else 0)
    df.gardenin = df.gardenin.apply(lambda x: 1 if x == 'Y' else 0)
    df.homee = df.homee.apply(lambda x: 1 if x == 'Y' else 0)
    df.pcowners = df.pcowners.apply(lambda x: 1 if x == 'Y' else 0)
    df.pets = df.pets.apply(lambda x: 1 if x == 'Y' else 0)
    df.photo = df.photo.apply(lambda x: 1 if x == 'Y' else 0)
    df.stereo = df.stereo.apply(lambda x: 1 if x == 'Y' else 0)
    df.veterans = df.veterans.apply(lambda x: 1 if x == 'Y' else 0)
    df.walker = df.walker.apply(lambda x: 1 if x == 'Y' else 0)

    df.gender = df.gender.apply(lambda x: 'U' if x == '' else x)

    df.domain = df.domain.apply(lambda x: 'N4' if x == ' ' else x)
    df.domain1 = df.domain.apply(lambda x: x[0])
    df.domain2 = df.domain.apply(lambda x: int(x[1]))
    df.drop(['domain'], inplace=True, axis=1)

    return df


df_test = transformations(df_test)
df = transformations(df)


categorical_cols = ['title', 'state', 'mailcode', 'recinhse', 'recp3',
                    'recpgvg', 'recsweep', 'homeownr', 'child03', 'child07',
                    'child12', 'child18', 'income_range', 'gender', 'hit',
                    'solp3', 'solih', 'major', 'collect1', 'veterans', 'bible',
                    'catlg', 'homee', 'pets', 'cdplay', 'stereo', 'pcowners',
                    'photo', 'crafts', 'fisher', 'gardenin', 'boats', 'walker',
                    'kidstuff', 'cards', 'plates', 'pepstrfl', 'msa', 'adi',
                    'dma']

numeric_cols = ['mbcraft', 'mbgarden', 'mbbooks', 'mbcolect', 'magfaml', 'magfem',
                'magmale', 'pubgardn', 'pubculin', 'pubhlth', 'pubdoity',
                'pubnewfn', 'pubphoto', 'pubopp', 'malemili', 'malevet',
                'vietvets', 'wwiivets', 'localgov', 'stategov', 'fedgov',
                'pop901', 'pop902', 'pop903', 'pop90c1', 'pop90c2',
                'pop90c3', 'pop90c4', 'pop90c5', 'eth1', 'eth2', 'eth3', 'eth4',
                'eth5', 'eth6', 'eth7', 'eth8', 'eth9', 'eth10', 'eth11', 'eth12',
                'eth13', 'eth14', 'eth15', 'eth16', 'agec1', 'agec2', 'agec3',
                'agec4', 'agec5', 'agec6', 'agec7', 'hhage1', 'hhage2', 'hhage3',
                'hhn1', 'hhn2', 'hhn3', 'hhn4', 'hhn5', 'hhn6', 'marr1', 'marr2',
                'marr3', 'marr4', 'hhp1', 'hhp2', 'dw1', 'dw2', 'dw3', 'dw4', 'dw5',
                'dw6', 'dw7', 'dw8', 'dw9', 'hv1', 'hv2', 'hv3', 'hv4', 'hu1',
                'hu2', 'hu3', 'hu4', 'hu5', 'hhd1', 'hhd2', 'hhd3', 'hhd4', 'hhd5',
                'hhd6', 'hhd7', 'hhd8', 'hhd9', 'hhd10', 'hhd11', 'hhd12', 'hvp1',
                'hvp2', 'hvp3', 'hvp4', 'hvp5', 'hvp6', 'hur1', 'hur2', 'rhp1',
                'rhp2', 'rhp3', 'rhp4', 'hupa1', 'hupa2', 'hupa3', 'hupa4', 'hupa5',
                'hupa6', 'hupa7', 'rp1', 'rp2', 'rp3', 'rp4',
                'ic1', 'ic2', 'ic3', 'ic4', 'ic5', 'ic6', 'ic7', 'ic8', 'ic9',
                'ic10', 'ic11', 'ic12', 'ic13', 'ic14', 'ic15', 'ic16', 'ic17',
                'ic18', 'ic19', 'ic20', 'ic21', 'ic22', 'ic23', 'hhas1', 'hhas2',
                'hhas3', 'hhas4', 'mc1', 'mc2', 'mc3', 'tpe1', 'tpe2', 'tpe3',
                'tpe4', 'tpe5', 'tpe6', 'tpe7', 'tpe8', 'tpe9', 'pec1', 'pec2',
                'tpe10', 'tpe11', 'tpe12', 'tpe13', 'lfc1', 'lfc2', 'lfc3', 'lfc4',
                'lfc5', 'lfc6', 'lfc7', 'lfc8', 'lfc9', 'lfc10', 'occ1', 'occ2',
                'occ3', 'occ4', 'occ5', 'occ6', 'occ7', 'occ8', 'occ9', 'occ10',
                'occ11', 'occ12', 'occ13', 'eic1', 'eic2', 'eic3', 'eic4', 'eic5',
                'eic6', 'eic7', 'eic8', 'eic9', 'eic10', 'eic11', 'eic12', 'eic13',
                'eic14', 'eic15', 'eic16', 'sec1', 'sec2', 'sec3', 'sec4', 'sec5',
                'afc1', 'afc2', 'afc3', 'afc4', 'afc5', 'afc6', 'vc1', 'vc2', 'vc3',
                'vc4', 'pobc1', 'pobc2', 'voc1', 'voc2', 'voc3', 'mhuc1', 'mhuc2',
                'ac1', 'ac2', 'cardprom', 'maxadate', 'numprom', 'cardpm12',
                'numprm12', 'ngiftall', 'cardgift', 'minramnt', 'minrdate',
                'maxramnt', 'maxrdate', 'lastgift', 'lastdate', 'fistdate',
                'nextdate', 'avggift', 'age', 'numchld', 'wealth1']


def convert_type(df):

    def string_to_numeric(x):
        try:
            return np.float(x)
        except Exception:
            return 0

    for col in numeric_cols:
        df[col] = df[col].apply(string_to_numeric)

    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df


df_test = convert_type(df_test)
df = convert_type(df)

features = categorical_cols + numeric_cols

# split test train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# doing pca on just numeric columns
df_train, df_val = train_test_split(df, test_size=0.4, random_state=1234)

x = df_train.loc[:, numeric_cols].values
y = df_val.loc[:, numeric_cols].values

scaler = StandardScaler()
scaler.fit(x)

x = scaler.transform(x)
y = scaler.transform(y)

pca = PCA(.95)
pca.fit(x)

x = pca.transform(x)

print(x.shape)
print(pca.n_components_)

plt.semilogy(pca.explained_variance_ratio_, '--o')


#handling categorical data
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_train.loc[:, categorical_cols].values)


#Joining function for marketing cost with test and train datasets
def zip_code_join(df, df_z):

    df.zip = df.zip.apply(lambda x: x.replace('-', ''))
    df = df.join(df_z, how='outer', on='zip')

    return df

