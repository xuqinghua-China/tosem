import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import datetime
from settings import Config
from utils import find_diff
from scipy.io import arff


def process_SWaT(config):
    # read
    fpath = config.swat_path
    pickle_path = config.pickle_path

    # process
    df = pd.read_excel(fpath, skiprows=1)
    col_names = [clean_swat_col_name(col_name) for col_name in df.columns]
    df.columns = col_names
    df = df.drop(columns=["Timestamp"])
    mappings = {
        "Normal/Attack": {
            "Normal": 0,
            "Attack": 1,
            "A ttack": 1
        }
    }
    df.replace(mappings, inplace=True)
    df.astype({"Normal/Attack": "int64"})
    # normalization
    data_columns = [col for col in df.columns if col != "Normal/Attack"]
    scaler = MinMaxScaler()
    df[data_columns] = scaler.fit_transform(df[data_columns])
    # ignore first 16000 records
    df = df.iloc[16001:]
    # train/test split
    split = int(config.train_test_ratio * df.shape[0])
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    # save
    pickle.dump(train_df, open(os.path.join(pickle_path, "swat_train.pkl"), "wb"))
    pickle.dump(test_df, open(os.path.join(pickle_path, "swat_test.pkl"), "wb"))
    # attack log
    attack_log=get_attack_log(df)
    pickle.dump(attack_log,open(config.swat_attack_log_path,"wb"))

def process_WADI(config):
    # read
    fpath = config.wadi_path
    pickle_path = config.pickle_path
    # process
    df = pd.read_csv(fpath)
    col_names = [clean_wadi_col_name(col_name) for col_name in df.columns]
    df.columns = col_names
    df["Normal/Attack"] = df.apply(lambda row: get_wadi_label(row), axis=1)
    print(df.head())
    df = df.drop(columns=["Date", "Time", "Row"])
    df.astype({"Normal/Attack": "int64"})
    # train/test split
    split = int(config.train_test_ratio * df.shape[0])
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    # save
    pickle.dump(train_df, open(os.path.join(pickle_path, "wadi_train.pkl"), "wb"))
    pickle.dump(test_df, open(os.path.join(pickle_path, "wadi_test.pkl"), "wb"))
    # attack log
    attack_log=get_attack_log(df)
    pickle.dump(attack_log,open(config.wadi_attack_log_path,"wb"))

def process_BATADAL(config):
    # read
    fpath = config.batadal_path
    pickle_path = config.pickle_path
    # process
    df = pd.read_csv(fpath, encoding="utf_8_sig")
    col_names = [clean_batadal_col_name(col_name) for col_name in df.columns]
    df.columns = col_names[:-1] + ["Normal/Attack"]
    print(df.columns)
    print(df.head())
    df = df.drop(columns=["datetime"])
    df.astype({"Normal/Attack": "int64"})
    # train/test split
    split = int(config.train_test_ratio * df.shape[0])
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    # save
    pickle.dump(train_df, open(os.path.join(pickle_path, "batadal_train.pkl"), "wb"))
    pickle.dump(test_df, open(os.path.join(pickle_path, "batadal_test.pkl"), "wb"))
    # attack log
    attack_log=get_attack_log(df)
    pickle.dump(attack_log,open(config.batadal_attack_log_path,"wb"))

def process_PHM(config):
    fpath = config.phm_path
    pickle_path = config.pickle_path
    # process
    df = pd.read_csv(fpath, skiprows=1, header=None)
    del df[df.columns[0]]
    changed_cols = [i for i in range(len(df.columns))]
    changed_cols[-1] = "Normal/Attack"
    df.columns = changed_cols
    df["Normal/Attack"] = df["Normal/Attack"].apply(lambda x: map_phm_label(x))
    # train_test_split
    split = int(config.train_test_ratio * df.shape[0])
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    # save
    pickle.dump(train_df, open(os.path.join(pickle_path, "phm_train.pkl"), "wb"))
    pickle.dump(test_df, open(os.path.join(pickle_path, "phm_test.pkl"), "wb"))
    # attack log
    attack_log = get_attack_log(df)
    pickle.dump(attack_log, open(config.phm_attack_log_path, "wb"))

def process_GAS(config):
    fpath = config.phm_path
    pickle_path = config.pickle_path
    # process
    data = arff.loadarff(fpath)
    df = pd.DataFrame(data[0])
    cols = df.columns
    cols[-1] = "Normal/Attack"
    df["Normal/Attack"] = df["Normal/Attack"].apply(lambda x: map_gas_label(x))
    # train_test_split
    split = int(config.train_test_ratio * df.shape[0])
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    # save
    pickle.dump(train_df, open(os.path.join(pickle_path, "phm_train.pkl"), "wb"))
    pickle.dump(test_df, open(os.path.join(pickle_path, "phm_test.pkl"), "wb"))
    # attack log
    attack_log = get_attack_log(df)
    pickle.dump(attack_log, open(config.gas_attack_log_path, "wb"))

# process label
# process saclar

def map_phm_label(x):
    if x == 1:
        return 0
    else:
        return 1


def map_gas_label(x):
    if x == 0:
        return 0
    else:
        return 1


def get_attack_log(df):
    i = 0
    log = []
    while i < df.size:
        if df.iloc[i]["Attack/Normal"] == 1:
            start_i = i
            while df.iloc[i]["Attack/Normal"] == 1:
                i += 1
            end_i = i
            log.append((start_i, end_i))
            i += 1


def clean_swat_col_name(name):
    return name.strip()


def clean_wadi_col_name(name):
    name = name.strip()
    splits = name.split("\\")
    return splits[-1]


def get_wadi_attack_times():
    attack_list = [
        (datetime.datetime(2017, 10, 9, 19, 25, 00), datetime.datetime(2017, 10, 9, 19, 50, 16)),
        (datetime.datetime(2017, 10, 10, 10, 24, 10), datetime.datetime(2017, 10, 10, 10, 34, 00)),
        (datetime.datetime(2017, 10, 10, 10, 55, 00), datetime.datetime(2017, 10, 10, 11, 24, 00)),
        (datetime.datetime(2017, 10, 10, 11, 30, 40), datetime.datetime(2017, 10, 10, 11, 44, 50)),
        (datetime.datetime(2017, 10, 10, 13, 39, 30), datetime.datetime(2017, 10, 10, 13, 50, 40)),
        (datetime.datetime(2017, 10, 10, 14, 48, 17), datetime.datetime(2017, 10, 10, 14, 59, 55)),
        (datetime.datetime(2017, 10, 10, 17, 40, 00), datetime.datetime(2017, 10, 10, 17, 49, 40)),
        (datetime.datetime(2017, 11, 10, 10, 55, 00), datetime.datetime(2017, 11, 10, 10, 56, 27)),
        (datetime.datetime(2017, 11, 10, 11, 17, 54), datetime.datetime(2017, 11, 10, 11, 31, 20)),
        (datetime.datetime(2017, 11, 10, 11, 36, 31), datetime.datetime(2017, 11, 10, 11, 47, 00)),
        (datetime.datetime(2017, 11, 10, 11, 59, 00), datetime.datetime(2017, 11, 10, 12, 5, 00)),
        (datetime.datetime(2017, 11, 10, 12, 7, 30), datetime.datetime(2017, 11, 10, 12, 10, 52)),
        (datetime.datetime(2017, 11, 10, 12, 16, 00), datetime.datetime(2017, 11, 10, 12, 25, 36)),
        (datetime.datetime(2017, 11, 10, 15, 26, 30), datetime.datetime(2017, 11, 10, 15, 37, 00)),
    ]
    return attack_list


def get_wadi_date(d):
    splits = d.split("/")
    return int(splits[0]), int(splits[1]), int(splits[2])


def get_wadi_time(t):
    splits = t.split(":")
    return int(splits[0]), int(splits[1]), int(splits[2][:2])


def get_wadi_label(row):
    attack_list = get_wadi_attack_times()
    dt = row["Date"]
    tm = row["Time"]
    day, month, year = get_wadi_date(dt)
    h, m, s = get_wadi_time(tm)
    timestamp = datetime.datetime(year, month, day, h, m, s)
    for start_time, end_time in attack_list:
        if start_time <= timestamp <= end_time:
            return 1
    else:
        return 0


def get_batadal_encoding(config):
    encoding_list = ['ascii', 'big5', 'big5hkscs', 'cp037', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737'
        , 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862'
        , 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949', 'cp950'
        , 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254'
        , 'cp1255', 'cp1256', 'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr'
        , 'gb2312', 'gbk', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2'
        , 'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1', 'iso8859_2'
        , 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7', 'iso8859_8', 'iso8859_9'
        , 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab'
        , 'koi8_r', 'koi8_t', 'koi8_u', 'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2'
        , 'mac_roman', 'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213', 'utf_32'
        , 'utf_32_be', 'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7', 'utf_8', 'utf_8_sig']

    for encoding in encoding_list:
        worked = True
        try:
            df = pd.read_csv(config.batadal_path, encoding=encoding, nrows=5)
        except:
            worked = False
        if worked:
            print(encoding, ':\n', df.head())


def clean_batadal_col_name(name):
    name = name.strip()
    name = name.lower()
    return name


def get_swat_col(config):
    data = pickle.load(open(os.path.join(config.pickle_path, "swat_train.pkl"), "rb"))
    cols = data.columns
    other_cols = [col for col in cols if "IT" in col]
    idx_cols = [col for col in cols if is_valid_swat_idx_col(col)]
    print(cols)
    print(len(other_cols), len(idx_cols), len(cols))
    assert len(other_cols) + len(idx_cols) + 1 == len(cols), find_diff(other_cols + idx_cols, cols)
    swat_idx = data[idx_cols]
    swat_idx = swat_idx.astype(int)
    print(swat_idx.head())
    pickle.dump(swat_idx, open(config.swat_idx_path, "wb"))

    indices = [i for i in range(len(cols)) if cols[i] in idx_cols]
    pickle.dump(indices, open(config.swat_idx_indices_path, "wb"))


def is_valid_swat_idx_col(col):
    valid_prefix = ["P", "UV", "MV"]
    for p in valid_prefix:
        if col.startswith(p) and "IT" not in col:
            return True

    else:
        return False


def get_wadi_col(config):
    data = pickle.load(open(os.path.join(config.pickle_path, "wadi_train.pkl"), "rb"))
    cols = data.columns
    suffixes = []
    for col in cols:
        splits = col.split("_")
        suffixes.append(splits[-1])
    print(set(suffixes))
    other_cols = [col for col in cols if is_valid_wadi_other_col(col)]
    idx_cols = [col for col in cols if is_valid_wadi_idx_col(col)]
    assert len(other_cols) + len(idx_cols) + 1 == len(cols), find_diff(other_cols + idx_cols, cols)
    wadi_idx = data[idx_cols]
    wadi_idx = wadi_idx.fillna(0)
    wadi_idx = wadi_idx.astype(int)
    print(wadi_idx.head())
    pickle.dump(wadi_idx, open(config.wadi_idx_path, "wb"))
    indices = [i for i in range(len(cols)) if cols[i] in idx_cols]
    pickle.dump(indices, open(config.wadi_idx_indices_path, "wb"))


def is_valid_wadi_idx_col(name):
    valid_suffix = ["STATUS", "AL", "AH", "LOG"]
    suffix = name.split("_")[-1]
    return suffix in valid_suffix


def is_valid_wadi_other_col(name):
    valid_other_suffix = ["CO", "FLOW", "PRESSURE", "SPEED", "PV", "SP"]
    suffix = name.split("_")[-1]
    return suffix in valid_other_suffix


def clean_wadi_name(names):
    pass


def get_batadal_col():
    data = pickle.load(open(os.path.join(config.pickle_path, "batadal_train.pkl"), "rb"))
    cols = data.columns
    suffixes = []
    for col in cols:
        splits = col.split("_")
        suffixes.append(splits[-1])
    other_cols = [col for col in cols if is_valid_batadal_other_col(col)]
    idx_cols = [col for col in cols if is_valid_batadal_idx_col(col)]
    assert len(other_cols) + len(idx_cols) + 1 == len(cols), find_diff(other_cols + idx_cols, cols)
    batadal_idx = data[idx_cols]
    batadal_idx = batadal_idx.fillna(0)
    batadal_idx = batadal_idx.astype(int)
    print(batadal_idx.head())
    pickle.dump(batadal_idx, open(config.batadal_idx_path, "wb"))
    indices = [i for i in range(len(cols)) if cols[i] in idx_cols]
    pickle.dump(indices, open(config.batadal_idx_indices_path, "wb"))


def is_valid_batadal_idx_col(col):
    valid_list = [
        "s_pu1", "s_pu2", "f_pu3", "s_pu3", "s_pu4", "s_pu5", "f_pu6", "s_pu6", "s_pu7", "s_pu8", "f_pu9", "s_pu9",
        "s_pu10", "s_pu11", "f_pu11", "s_v2", "f_pu5"
    ]
    return col in valid_list


def is_valid_batadal_other_col(col):
    valid_list = [
        "l_t1", "l_t2", "l_t3", "l_t4", "l_t5", "l_t6", "l_t7", "f_pu1", "f_pu2", "f_pu4", "f_pu7", "f_pu8", "f_pu10",
        "f_v2", "p_j280", "p_j269", "p_j300", "p_j256", "p_j289", "p_j415", "p_j302", "p_j306", "p_j307", "p_j317",
        "p_j14", "p_j422"
    ]
    return col in valid_list




if __name__ == '__main__':
    config = Config()
    # process_SWaT(config)
    # process_WADI(config)
    # get_batadal_encoding(config)
    # process_BATADAL(config)
    # get_swat_col(config)
    # get_wadi_col(config)
    get_batadal_col()
