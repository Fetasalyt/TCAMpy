Theoretical Background
======================

The theoretical background for this model is based on the work of Carlos A Valentim, José A Rabi and Sergio A David. I expanded this model by simulating an immune response during the growth of the tumor cells,
as well as random mutations, influencing tumor survival. Other ideas, like nutrition may also be implemented in the future.

Valentim CA, Rabi JA, David SA. Cellular-automaton model for tumor growth dynamics: Virtualization of different scenarios.
Comput Biol Med. 2023 Feb;153:106481. doi: 10.1016/j.compbiomed.2022.106481. Epub 2022 Dec 28. PMID: 36587567.
(url: https://pubmed.ncbi.nlm.nih.gov/36587567/)

This is a 2D cellular automata model; the area is a square grid, with each cell being able to have discrate values ranging from 0 to a 'pmax'+1 (see below). If a square
is 0, it is empty, if not 0, a tumor cell exists there with a proliferation potential of the square's value. All probabilites mentioned below, the maximum proliferation potential value,
model duration and area size is a parameter of the model, which can be set by the user. Here is a quick explanation of how the model works.

Tumor cell types
----------------

In the model, we have two types of tumor cells. RTCs (regular tumor cells) can only divide a limited amount of times. This amount is their proliferation potential,
which gets smaller with every division. These cells can only proliferate into other RTCs, and the daughter cell will have the same proliferation potential value as the
mother cell after the division (So an RTC with pp = 5 will create two cells, whose pp = 4). STCs (stem tumor cells) have infinite divisions. Their daughter cell is either
an RTC with the highest proliferation potential value (pmax) or (by a small chance) another STC. The value which we represent these cells with in the model is pmax+1.

Tumor cell behaviour
--------------------

At every cycle each cell chooses from four different options. First, they can die by apoptosis (low chance for RTCs only). If they survive, they can proliferate with the probability
of CCT*dt/24 (where CCT is the cell cycle time, dt is the time step in the model). In case of no proliferation, they can migrate to one of the eight neighbouring cells. The probability
of this action is mu*dt (mu is the migration capacity of the cell). If none of these actions happen, the cell stays quiescent. Quiescence is forced if there is no free space around the
a surviving cell.

Immune response
---------------

In each cycle, after the tumor growth, immune cells may also appear and act. They spawn randomly on the inner edge of the field and every time step they move to a free neighbouring spot. If they get in contact with a cancer cell (they move next to one), they might kill it. After a certain amount of life, the immune cells die. The spawn rate, kill chance and lifespan is influenced by a single ‘immune strength (I)’ parameter, and the spawn rate also increases with the size of the tumor (but has a maximum value based on I). Immune cells are shown on the same heatmap as the tumor cells. The immune response can be disabled by setting the 'I' parameter to 0.

Mutations
---------

With every proliferation the daughter cell can aquire a new mutation (+1 or -1 to their 'mutation state', up to +3 or -3), and inherit the mother cell's mutations. The mutation state influences proliferation rate, apoptosis chance, immune resistance and current proliferation potential.
A positive mutation value means mutations that help the cancer cell survive, while a negative one is bad for the tumor. For example, a cancer cell with a mutation state of +2 will proliferate with a higher then normal chance, and is less likely to die. At every proliferation it's daughter cell
can either stay a +2 or gain another positive mutation to +3 or a negative one to +1. The chance for this is the 'M' parameter of the model. A mutation map, containg what mutations happened where, is shown on a 'Mutation history' heatmap. The immune response can be disabled by setting the 'M' parameter to 0.

Machine Learning
----------------

Basic Machine Learning features are implemented in the module with sklearn. These include generating a dataset with randomized parameter values (with ranges set by the user), training a model and calculating the most important parameters for tumor growth,
as well as predicting final tumor size or confluence based on a new set of parameters. Additional features and improvements are also planned for later.
