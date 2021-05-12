import getopt
import sys

from colorama import Fore
import warnings

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.cgan.Cgan import Cgan
from models.dcgan.Dcgan import Dcgan
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd
from models.infogan.Infogan import Infogan

separatorStr = "\n***************************************************************\n"
beginMsg = "******** Beginning Training ********"
completeMsg = "******** Completed Training ********\n"


def get_updated_file_name(dir_loc, file_name, gan_name, training, ext, sep):
    return dir_loc + file_name + sep + gan_name + sep + training + ext


def set_gan(gan_name, training, dir_loc, epoch):
    gans = dict()
    gans['seqgan'] = Seqgan
    gans['gsgan'] = Gsgan
    gans['textgan'] = TextganMmd
    gans['leakgan'] = Leakgan
    gans['rankgan'] = Rankgan
    gans['maligan'] = Maligan
    gans['cgan'] = Cgan
    gans['dcgan'] = Dcgan
    gans['infogan'] = Infogan
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        gan.vocab_size = 5000
        gan.generate_num = 10000
        gan.oracle_file = get_updated_file_name(dir_loc, 'oracle', gan_name, training, '.txt', '_')
        gan.generator_file = get_updated_file_name(dir_loc, 'generator', gan_name, training, '.txt', '_')
        gan.test_file = get_updated_file_name(dir_loc, 'test_file', gan_name, training, '.txt', '_')
        gan.log_file = get_updated_file_name(dir_loc, 'experiment-log', gan_name, training, '.csv', '-')
        gan.pre_epoch_num = epoch
        gan.adversarial_epoch_num = epoch
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)


def set_training(gan, training_method):
    print("set training")
    print(training_method)
    try:
        if training_method == 'oracle':
            gan_func = gan.train_oracle
        elif training_method == 'cfg':
            gan_func = gan.train_cfg
        elif training_method == 'real':
            gan_func = gan.train_real
        else:
            print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
            # sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        # sys.exit(-3)
    return gan_func


def parse_cmd(argv):
    try:
        argvals = ' '.join(argv)
        if argvals == '':
            print(beginMsg)
            gan = None
            # add all trainings to array
            trainings = ["oracle", "cfg", "real"]
            dir_loc = 'results/'
            epoch = 5;
            opt_arg = {}
            key = "-g"
            # check if key is already present in dict
            if key not in opt_arg:
                opt_arg[key] = []

            # add all modes to -g flag
            opt_arg["-g"].append('seqgan')
            opt_arg["-g"].append('rankgan')
            opt_arg["-g"].append('mle')
            opt_arg["-g"].append('maligan')
            opt_arg["-g"].append('leakgan')
            opt_arg["-g"].append('textgan')
            opt_arg["-g"].append('gsgan')
            opt_arg["-g"].append('cgan')
            opt_arg["-g"].append('dcgan')
            opt_arg["-g"].append('infogan')

            print('Results output directory location set: ' + dir_loc)
            print('Epochs Set: ' + str(epoch))

            for training in trainings:
                print("try with training.." + training)
                for value in opt_arg.values():
                    for ganName in value:
                        try:
                            print("training GAN..." + ganName)
                            try:
                                gan = set_gan(ganName, training, dir_loc, epoch)
                            except Exception as e:
                                print("setGan exception")
                                print(e)
                            print("start training GAN..." + ganName)
                            try:
                                gan.train_oracle()
                            except Exception as e:
                                print("Training exception")
                                print(e)
                            print("GAN function")
                            try:
                                gan_func = set_training(gan, training)
                            except Exception as e:
                                print("Gan Function exception1")
                                print(e)
                            print("Run")
                            try:
                                gan_func()
                            except Exception as e:
                                print("Gan Function exception2")
                                print(e)

                        except Exception as e:
                            print("Main exception1")
                            print(e)
                        print(separatorStr)
            print(completeMsg)
        else:
            print(beginMsg)
            gan = None
            opts, args = getopt.getopt(argv, "hg:t:d:o:p:")
            opt_arg = dict(opts)
            if '-h' in opt_arg.keys():
                print('usage: python main.py -g <gan_type>')
                print('       python main.py -g <gan_type> -t <train_type>')
                print('       python main.py -g <gan_type> -t realdata -d <your_data_location>')
                print('       python main.py -g <gan_type> -t realdata -d <your_data_location> -o <output_dir_for_results>')
                print('       python main.py -g <gan_type> -t realdata -d <your_data_location> -o <output_dir_for_results> -p <epoch_num>')

                sys.exit(0)
            
            training = None
            if '-t' in opt_arg.keys():
                training = opt_arg['-t']
            else:
                print('Unspecified Training Type: Defaulting to Oracle')
                training = 'oracle'

            dir_loc = None
            # get output directory for results if specified
            if '-o' in opt_arg.keys():
                dir_loc = opt_arg['-o']
                if not dir_loc.endswith('/'):
                    dir_loc += '/'
            else:
                print('Unspecified Output Director Location: Defaulting to --> results/')
                dir_loc = 'results/'

            epoch = None
            # get epoch value to use if specified
            if '-p' in opt_arg.keys():
                temp = opt_arg['-p']
                if(temp.isdigit()):
                    epoch = int(temp)
                else:
                    print('Invalid Epoch Argument Passed: Defaulting Epoch to 5')
                    epoch = 5;
            else:
                print('Unspecified Epoch Setting: Defaulting Epoch to 5')
                epoch = 5;

            model = None
            if not '-g' in opt_arg.keys():
                print('Unspecified GAN Type: Defaulting to MLE Model')
                model = 'mle'
            else:
                model = opt_arg['-g']
                
            gan = set_gan(model, training, dir_loc, epoch)
            
            data_loc = None
            if '-d' in opt_arg.keys():
                data_loc = opt_arg['-d']
            else:
                print('Unspecified Data Set: Defaulting to image_coco')
                data_loc = 'data/image_coco.txt'

            print('\n    <--        Metrics Set           -->')
            print('Model Set:                    '  + model.capitalize())
            print('Training Set:                 ' + training.capitalize())
            print('Data Set:                     ' + data_loc)
            print('Epochs Set:                   ' + str(epoch))
            print('Results Output Directory Set: ' + dir_loc)
            print('    <--          End Details             -->\n')
            sys.exit(-1)
            if not '-t' in opt_arg.keys():
                gan.train_oracle()
            else:
                gan_func = set_training(gan, training)
                if training == 'real' and '-d' in opt_arg.keys():
                    gan_func(data_loc)
                else:
                    gan_func()

            print(completeMsg)

    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass


if __name__ == '__main__':
    # Init the flags so models dependent on flags do not break with arg use
    flags = tf.compat.v1.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('g', "", 'Default g')
    flags.DEFINE_string('t', "", 'Default t')
    flags.DEFINE_string('d', "", 'Default d')
    flags.DEFINE_string('o', "", 'Default o')
    flags.DEFINE_string('p', "", 'Default p')
    # parse the command
    parse_cmd(sys.argv[1:])
