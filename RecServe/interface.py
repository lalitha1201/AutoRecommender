import click
from colorama import Fore,Style
import data_preprocessing as data
import model_cf as model


def colored(string,color):
    return color + string+Fore.RESET

class User_Input:
    def __init__(self,gender,age,occupation,location):
        self.gender = gender
        self.age = age
        self.occupation = occupation
        self.location = location

        self.weights= {}
        self.weights['age'] = 0.25
        self.weights['gender'] = 0.25
        self.weights['occupation'] = 0.25
        self.weights['location'] = 0.25

    def set_weights(self,key,values):
        self.weights[key]=values

@click.command()
def dialogue():
    click.echo('Welcome to AutoRec! Let me help you with the product recommendations.')
    name = click.prompt(colored('Enter your name',Fore.MAGENTA))
    #click.echo(name)
    csv = click.prompt(colored('Enter the path for your sample data',Fore.MAGENTA))
    #click.echo(name)
    click.clear()
    #user = click.prompt(colored('do you want to recommend items for users to purchase? [Y/N]',Fore.GREEN), type= click.Choice(['Y','N']))
    #csv = click.prompt(colored('Enter the path',Fore.MAGENTA))
    #click.echo(name)
    click.clear()
    modelname = 'user_user'
    data_subset = data.get_data(csv)
    location = click.echo('Data being preprocessed')
    modelname = click.prompt(colored('Do you want to recommend items for users to purchase? [y/N]"',Fore.MAGENTA),type= click.Choice(['Y','N']))
    #click.echo(name)
    click.clear()
    results = model.all_models(csv,modelname)
    modeled = click.echo('Data being modeled')
    modelname1 = click.prompt(colored('Do you want to recommend similar items for users to purchase? [y/N]"',Fore.MAGENTA),type= click.Choice(['Y','N']))
    #click.echo(name)
    click.clear()
    modelname = 'item_item'
    results = model.all_models(csv,modelname)
    
    
    


if __name__ == '__main__':
    dialogue()
