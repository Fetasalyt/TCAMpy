# TCAMpy

This is a single python module for a cellular automaton modeling tumor growth. The user can set the parameters, create unique initial states, view and save statistics, save data and also use a streamlit dashboard as a graphical interface. Growth plots, histograms and animation is available for visualization in an easy to use way.

The theoretical background for this model is based on the work of Carlos A Valentim, José A Rabi and Sergio A David. I expanded this model by simulating an immune response during the growth of the tumor cells. Other ideas, like mutations and nutrition may also be implemented in the future.

Valentim CA, Rabi JA, David SA. Cellular-automaton model for tumor growth dynamics: Virtualization of different scenarios. Comput Biol Med. 2023 Feb;153:106481. doi: 10.1016/j.compbiomed.2022.106481. Epub 2022 Dec 28. PMID: 36587567.
(url: https://pubmed.ncbi.nlm.nih.gov/36587567/)

This documentation provides detaild description on how to use the modul, with example codes and links to example files.

## Example Results

Visualization from running a single model. If enabled, the user recieves an image of the growth a line graph of cell numbers over time and a histogram of proliferation potentials.

![Plot example](https://github.com/Fetasalyt/TCAMpy/blob/main/Images/single_example.png?raw=true)

Animation is available to turn on when running the model as well. Visualizing a dataframe containing results from multiple model executions is also possible. A cell number line graph of the average numbers
and a histogram of the average proliferation potentials can be plotted with standard deviations.

![Averages plot example](https://github.com/Fetasalyt/TCAMpy/blob/main/Images/avg_example.png?raw=true)

---

## Links

- [Homepage](https://github.com/FetasaLYt/TCAMpy)
- [Documentation](https://tcampy.readthedocs.io/en/latest/)
- [Repository](https://github.com/FetasaLYt/TCAMpy)
- [Issues](https://github.com/FetasaLYt/TCAMpy/issues)
