import getopt
import sys

from colorama import Fore

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.cgan.Cgan import Cgan
from models.Dcgan.Dcgan import Dcgan
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd


separatorStr = "\n***************************************************************\n"
beginMsg = "******** Beginning Training ********"
completeMsg = "******** Completed Training ********\n"

def set_gan(gan_name):
    gans = dict()
    gans['seqgan'] = Seqgan
    gans['gsgan'] = Gsgan
    gans['textgan'] = TextganMmd
    gans['leakgan'] = Leakgan
    gans['rankgan'] = Rankgan
    gans['maligan'] = Maligan
    gans['cgan'] = Cgan
    #gans['2_Gan'] = 2_Gan
    gans['dcgan'] = Dcgan
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        gan.vocab_size = 5000
        gan.generate_num = 10000
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
            #sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        #sys.exit(-3)
    return gan_func


def parse_cmd(argv):
    try:
        argvals = ' '.join(argv)
        if argvals == '':
            print(beginMsg)
            gan = None
            trainings = ["oracle", "cfg", "real"]
            #parse_cmd(sys.argv[1:])
            # parse_cmd(sys.argv[1:])
            opt_arg = {}
            key = "-g"
            # check if key is already present in dict
            if key not in opt_arg:
                opt_arg[key] = []


            opt_arg["-g"].append('seqgan')
            opt_arg["-g"].append('rankgan')
            opt_arg["-g"].append('mle')
            opt_arg["-g"].append('maligan')
            opt_arg["-g"].append('leakgan')
            opt_arg["-g"].append('textgan')
            opt_arg["-g"].append('gsgan')
            opt_arg["-g"].append('cgan')
            #opt_arg["-g"].append('2_Gan')
            opt_arg["-g"].append('dcgan')
            for training in trainings:
             print("try with training.." + training)
             for value in opt_arg.values():
              for ganName in value:
               try:
                  print("training GAN..."+ ganName)
                  try:
                   gan = set_gan(ganName)
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
            opts, args = getopt.getopt(argv, "hg:t:d:")
            opt_arg = dict(opts)
            if '-h' in opt_arg.keys():
                print('usage: python main.py -g <gan_type>')
                print('       python main.py -g <gan_type> -t <train_type>')
                print('       python main.py -g <gan_type> -t realdata -d <your_data_location>')
                sys.exit(0)
            if not '-g' in opt_arg.keys():
                print('unspecified GAN type, use MLE training only...')
                gan = set_gan('mle')
            else:
                gan = set_gan(opt_arg['-g'])
            if not '-t' in opt_arg.keys():
                gan.train_oracle()
            else:
                gan_func = set_training(gan, opt_arg['-t'])
                if opt_arg['-t'] == 'real' and '-d' in opt_arg.keys():
                    gan_func(opt_arg['-d'])
                else:
                    gan_func()
                    
            print(completeMsg)
            
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass
    
    
if __name__ == '__main__':
  parse_cmd(sys.argv[1:])
  