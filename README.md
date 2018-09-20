# 2018 IML challenge 

The challenge detail are decribed in the wiki. To submit a solution (no later than 10th of April 24:00 CERN time) please make a PR request of a new file in the solutions directory. The commit comment should include your name and the file should be either numpy or root.

Task: Regress the soft-drop mass of jets with high transverse momentum. Jets are complex physical objects, often containing a spray of particles.https://en.wikipedia.org/wiki/Jet_(particle_physics)


## This solution won the 3rd prize in the above challenge

The pipeline is illustrated in detail in the notebooks above:

preprocess --> train_autoencoder --> train_regressor --> evaluation


The process involves training an autoencoder to condense and bring the jet constituent features into a constant length code , then concatenate it with reconstructed jet features, and train a neural net supervised regressor.
