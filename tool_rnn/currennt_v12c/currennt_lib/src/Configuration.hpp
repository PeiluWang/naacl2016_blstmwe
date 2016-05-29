/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "Types.hpp"

#include <string>


/******************************************************************************************//**
 * Creates the configuration for the program from the command line
 *********************************************************************************************/
class Configuration
{
public:
    enum optimizer_type_t {
        OPTIMIZER_STEEPESTDESCENT,
        OPTIMIZER_RPROP
    };

    enum distribution_type_t {
        DISTRIBUTION_NORMAL,
        DISTRIBUTION_UNIFORM
    };

private:
    static Configuration *ms_instance;

    std::string m_serializedOptions;

    bool m_trainingMode;
    bool m_hybridOnlineBatch;
    bool m_useCuda;
    bool m_shuffleFractions;
    bool m_shuffleSequences;
    bool m_autosave;

    optimizer_type_t    m_optimizer;
    distribution_type_t m_weightsDistribution;

    unsigned m_parallelSequences;
    unsigned m_maxEpochs;
    unsigned m_maxEpochsNoBest;
    unsigned m_validateEvery;
    unsigned m_testEvery;
    unsigned m_randomSeed;

    real_t m_learningRate;
    real_t m_momentum;
    real_t m_weightsUniformMin;
    real_t m_weightsUniformMax;
    real_t m_weightsNormalSigma;
    real_t m_weightsNormalMean;
    real_t m_inputNoiseSigma;
    real_t m_trainingFraction;
    real_t m_validationFraction;
    real_t m_testFraction;

    std::string m_networkFile;
    std::string m_trainingFile;
    std::string m_validationFile;
    std::string m_testFile;
    std::string m_trainedNetwork;
    std::string m_feedForwardInputFile;
    std::string m_feedForwardOutputFile;
    std::string m_autosavePrefix;
    std::string m_continueFile;

	// the device id of gpu device to lock
	int m_gpu_deviceid;
	// the dir of tmp cache file
	std::string m_cachedir;
	// 0 use the continue model's cfg, 1 use user's arg option
	int m_continue_cfgsrc;
	// the random seed used for frac shuffle
	unsigned m_randomseed_shuffle;
	// welyaer
	int n_vocab_size;
	int n_we_dim;
	int n_inputfeat_dim;
	// true, sort traindata (default); false, not sort
	bool m_sortTrainData;
	// file to save weweights
	std::string s_save_weweightsFile;
	// file to load weweights
	std::string s_load_weweightsFile;
	// file to save featweights
	std::string s_save_featweightsFile;
	// file to load featweights
	std::string s_load_featweightsFile;
	// file to load lstmlayer weights
	std::string s_load_lstmweightsFile;

public:
    /**
     * Parses the command line
     *
     * @param argc Number of strings in argv
     * @param argv Strings from the command line
     */
    Configuration(int argc, const char *argv[]);

    /**
     * Destructor
     */
    ~Configuration();

    /**
     * Returns the static instance
     *
     * @return The static instance
     */
    static const Configuration& instance();

    /**
     * Returns a string that contains all options
     * 
     * @return A string that contains all options
     */
    const std::string& serializedOptions() const;

    /**
     * Returns true if the NN shall be trained
     *
     * @return True if the NN shall be trained
     */
    bool trainingMode() const;

    /**
     * Returns true if hybrid online/batch learning is enabled
     *
     * Hybrid online/batch learning means that the network weights are updated after every
     * block of parallel sequences. Example: if the user sets parallel_sequences=50, then
     * the weights are updated after each block of 50 sequences has been processed.
     *
     * If the number of parallel sequences is set to 1, we have true online learning with
     * weight updates after every sequence.
     *
     * @return True if hybrid online/batch learning is enabled
     */
    bool hybridOnlineBatch() const;

    /**
     * Returns true if shuffling of fractions in hybrid online/batch learning is enabled
     *
     * Each fraction contains of N parallel sequences except for the last block which can
     * contain less sequences. If this option is enabled, the order in which the fractions
     * are computed is randomized for each training epoch.
     *
     * @return True if shuffling of fractions is enabled
     */
    bool shuffleFractions() const;

    /**
     * Returns true if shuffling of sequences within and across fractions is enabled
     *
     * If this option is enabled, the sequences are shuffled before each training epoch, 
     * resulting in a completely randomized distribution of sequences across the fractions.
     *
     * @return True if shuffling of sequences is enabled
     */
    bool shuffleSequences() const;

    /**
     * Returns true if the nVidia CUDA technology shall be used to accelerate the computations
     *
     * @return True if CUDA shall be used
     */
    bool useCuda() const;

    /**
     * Returns true if autosave is enabled
     *
     * @return True if autosave is enabled
     */
    bool autosave() const;

    /**
     * Returns the optimizer type
     *
     * @return The optimizer type
     */
    optimizer_type_t optimizer() const;

    /**
     * Returns the maximum number of parallel computed sequences
     *
     * @return The maximum number of parallel computed sequences
     */
    int parallelSequences() const;

    /**
     * Returns the maximum number of epochs during training
     *
     * @return The maximum number of epochs during training
     */
    int maxEpochs() const;

    /**
     * Returns the maximum number of training epochs in which no new lowest error could be achieved
     *
     * @return The maximum number of training epochs in which no new lowest error could be achieved
     */
    int maxEpochsNoBest() const;

    /**
     * Returns the number of training epochs after which the validation error shall be calculated
     *
     * @return The number of training epochs after which the validation error shall be calculated
     */
    int validateEvery() const;

    /**
     * Returns the number of training epochs after which the test error shall be calculated
     *
     * @return The number of training epochs after which the test error shall be calculated
     */
    int testEvery() const;

    /**
     * Returns the learning rate for the steepest descent optimizer
     *
     * @return The learning rate for the steepest descent optimizer
     */
    real_t learningRate() const;

    /**
     * Returns the momentum for the steepest descent optimizer
     *
     * @return The momentum for the steepest descent optimizer
     */
    real_t momentum() const;

    /**
     * Returns the path to the NN layout and weights file
     *
     * @return The path to the NN layout and weights file
     */
    const std::string& networkFile() const;

    /**
     * Returns the path to the *.nc file containing the training sequences
     *
     * @return The path to the *.nc file containing the training sequences
     */
    const std::string& trainingFile() const;

    /**
     * Returns the path to the *.nc file containing the validation sequences
     *
     * @return The path to the *.nc file containing the validation sequences
     */
    const std::string& validationFile() const;

    /**
     * Returns the path to the *.nc file containing the test sequences
     *
     * @return The path to the *.nc file containing the test sequences
     */
    const std::string& testFile() const;

    /**
     * Returns the seed for the random number generator
     *
     * @return The seed for the random number generator
     */
    unsigned randomSeed() const;

    /**
     * Returns the distribution type of the initial weights
     *
     * @return The distribution type of the initial weights
     */
    distribution_type_t weightsDistributionType() const;

    /**
     * Returns the minimum value of the uniform distribution of the initial weights
     *
     * @return The minimum value of the uniform distribution of the initial weights
     */
    real_t weightsDistributionUniformMin() const;

    /**
     * Returns the maximum value of the uniform distribution of the initial weights
     *
     * @return The maximum value of the uniform distribution of the initial weights
     */
    real_t weightsDistributionUniformMax() const;

    /**
     * Returns the sigma of the normal distribution of the initial weights
     *
     * @return The sigma of the normal distribution of the initial weights
     */
    real_t weightsDistributionNormalSigma() const;

    /**
     * Returns the mean of the normal distribution of the initial weights
     *
     * @return The mean of the normal distribution of the initial weights
     */
    real_t weightsDistributionNormalMean() const;

    /**
     * Returns the sigma of the normal distribution of the input noise
     *
     * @return The sigma of the normal distribution of the input noise
     */
    real_t inputNoiseSigma() const;

    /**
     * Returns the fraction of the training set to use
     *
     * @return The fraction of the training set to use
     */
    real_t trainingFraction() const;

    /**
     * Returns the validation of the training set to use
     *
     * @return The validation of the training set to use
     */
    real_t validationFraction() const;

    /**
     * Returns the test of the training set to use
     *
     * @return The test of the training set to use
     */
    real_t testFraction() const;

    /**
     * Returns the path to the trained network file
     *
     * @return The path to the trained network file
     */
    const std::string& trainedNetworkFile() const;

    /**
     * Returns the path to the feed forward input file
     *
     * @return The path to the feed forward input file
     */
    const std::string& feedForwardInputFile() const;

    /**
     * Returns the path to the feed forward output file
     *
     * @return The path to the feed forward output file
     */
    const std::string& feedForwardOutputFile() const;

    /**
     * Returns the autosave filename prefix
     *
     * @return The autosave filename prefix
     */
    const std::string& autosavePrefix() const;

    /**
     * Returns the autosave file from which training will continue
     *
     * @return The autosave file from which training will continue
     */
    const std::string& continueFile() const;

	bool lockgpudevice(int device_id);

	const std::string& cacheDir() const;

	unsigned randomSeedShuffle() const;

	int vocabSize() const;

	int inputWeDim() const;

	int inputFeatDim() const;

	bool sortTrainData() const;

	const std::string& saveWeweightsFile() const;

	const std::string& loadWeweightsFile() const;

	const std::string& saveFeatweightsFile() const;

	const std::string& loadFeatweightsFile() const;

	const std::string& loadLstmweightsFile() const;
};


#endif // CONFIGURATION_HPP
