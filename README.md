# Schnapsen platform - Project Intelligent Systems 2022-2023

## Getting started

This is the improved platform for the schnapsen card game. To get to know the concept of the game, please visit
[this web page](https://www.pagat.com/marriage/schnaps.html).


To use the platform, your python version must be at least 3.9, we suggest installing conda an using an environment.

To get started, install the schnapsen package and itse dependencies in editable mode by running:

```sh
pip install -e .
```

To run the tests, run:

```sh
pip install -e '.[test]'  # on Linux / MacOS
pip install -e ".[test]"  # on Windows
pytest ./tests
```

If the above fails, try deactivating your environment and activating it again.
Then retry installing the dependencies.

## Running the CLI

After intalling, you can try the provided command line interface examples.
Most examples are bots playing against each other; read the code for details.

To run the CLI, run:

```sh
python executables/cli.py
```
This will list the available commands.

For example, if you want try a RandBot play against another RandBot, type
`python executables/cli.py random-game`.


## Running the GUI

The graphical user interface (GUI) lets you play visually against a bot (e.g., You vs. RandBot).

To start the GUI, run:

```sh
python executables/server.py
```

Now, open your webbrowser and type in the server address (i.e., http://127.0.0.1:8080). 
By default, you are playing against RandBot. You can also play against other bots. Run 

```sh
python executables/server.py --help
```
for more details.

## Implementing more bots

You will find bot examples in the [`src/schnapsen/bots`](./src/schnapsen/bots) folder.
You can look at the example_bot.py file for various methods provided to your bot.


## Troubleshooting

### Getting the right python ###

The first hurdle in getting the platform to run is getting the right python version on your system.
An easy way to get that is using virtual environments. We suggest you install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to manage them.
Then, you can use conda to create a new environment by running
```sh
conda create --name project_is python=3.10
```
With this environment created, you can start it
```
conda activate project_is
```
Inside this environment you can install the dependencies as instructed above.

### Run the right python ###

If you install conda and create an environment, you can run python by just running the `python` command.
However, often your system also provides a python version. 
To know which python is running, use
```sh
which python    # on linux
where python    # on windows (untested)
``` 
Now, you want to look at the output and make sure that this executable is inside the anaconda folder and not where your system stores its executables.




<!--

Most of the time, when you read Github python repo READMEs, they won't tell you how to do things in detail, but simply tell you things like run `python bar`, run `pip install foo`, etc. All of these imply that you are running things in an isolated python environment. Often times this is easily done by creating virtual environments (e.g., venv, conda, etc.), where you know exactly what `python`, `pip`, and other modules you are running. If you are not familiar with it and still want to proceed on your current machine, especially on Windows, below are some tips.

1. **Be super specific with your python binary.**

   Don't just run `python bar` but do more like `python3.9 bar`. If you just run `python bar`, it's hard to know which python binary file your system is running.

2. **Be super specific with the modules (e.g., pip, pytest).**

   Don't just run `pip install foo` but do more like `python3.9 -m pip install foo`. Again, if you just run `pip install foo`, we don't know exactly which `pip` your system will run. `python3.9 -m pip install foo` specifies that you want your `python3.9` to run the module (i.e., `-m`) `pip` to do something. The same goes for `python3.9 -m pytest ./tests`, instead of `pytest ./tests`.

Things can be messy if you have multiple python3.9 versions (e.g., `python3.9.1`, `python3.9.10`, etc.). Things can get even more messy when your python binary can't be run as `python3.9` but more like `py3.9` or something. Good luck!
-->