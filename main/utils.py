import os
import json
import pickle
import logging
import configparser

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))
program_config_path = os.path.join(path, 'conf/settings.ini')
program_config_path = os.path.join(path, 'conf/settings_docker.ini')

# logging message
logging.basicConfig(level = logging.INFO,format ='[%(asctime)s : %(levelname)s/%(processName)s] %(message)s')
logger = logging.getLogger(__name__)

def get_config_values(section, option):
    config = configparser.ConfigParser()
    config.read(program_config_path)
    value = config.get(section=section, option=option)
    return value

def load_json(filename, mode='r'):
    data = []
    with open(filename, mode, encoding='utf-8') as fp:
        for line in fp.readlines():
            data.append(json.loads(line))
    return data

def load_pickle(filename, mode='rb'):
    with open(filename, mode) as fp:
        data = pickle.load(fp)
    return data

def save_pickle(filename, data, mode='wb'):
    with open(filename, mode) as fp:
        pickle.dump(data, fp)

if __name__ == '__main__':
    value = get_config_values('dataset', 'train')
    print (value)

