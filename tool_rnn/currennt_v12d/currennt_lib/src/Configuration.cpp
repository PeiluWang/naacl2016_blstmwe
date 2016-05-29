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

#include "Configuration.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filestream.h"

#include <limits>
#include <fstream>
#include <sstream>

#include <Windows.h>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/random/random_device.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>
namespace po = boost::program_options;

#define DEFAULT_UINT_MAX std::numeric_limits<unsigned>::max(), "inf"

Configuration *Configuration::ms_instance = NULL;


namespace internal {

std::string serializeOptions(const po::variables_map &vm) 
{
    std::string s;

    for (po::variables_map::const_iterator it = vm.begin(); it != vm.end(); ++it) {
        if (it->second.value().type() == typeid(bool))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<bool>(it->second.value()));
        else if (it->second.value().type() == typeid(unsigned))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<unsigned>(it->second.value()));
        else if (it->second.value().type() == typeid(float))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<float>(it->second.value()));
        else if (it->second.value().type() == typeid(double))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<double>(it->second.value()));
        else if (it->second.value().type() == typeid(std::string))
            s += it->first + '=' + boost::any_cast<std::string>(it->second.value());

        s += ";;;";
    }

    return s;
}

void deserializeOptions(const std::string &autosaveFile, std::stringstream *ss)
{
    // open the file
    std::ifstream ifs(autosaveFile.c_str(), std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Cannot open file");

    // calculate the file size in bytes
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    // read the file into a buffer
    char *buffer = new char[size + 1];
    ifs.read(buffer, size);
    buffer[size] = '\0';

    // parse the JSON file
    rapidjson::Document jsonDoc;
    if (jsonDoc.Parse<0>(buffer).HasParseError())
        throw std::runtime_error(std::string("Parse error: ") + jsonDoc.GetParseError());

    // extract the options
    if (!jsonDoc.HasMember("configuration"))
        throw std::runtime_error("Missing string 'configuration'");

    std::string s = jsonDoc["configuration"].GetString();
    (*ss) << boost::replace_all_copy(s, ";;;", "\n");
}

} // namespace internal


Configuration::Configuration(int argc, const char *argv[])
{
    if (ms_instance)
        throw std::runtime_error("Static instance of class Configuration already created");
    else
        ms_instance = this;

    std::string optionsFile;
    std::string optimizerString;
    std::string weightsDistString;

    // create the command line options
    po::options_description commonOptions("Common options");
    commonOptions.add_options()
        ("help",                                                                              "shows this help message")
        ("options_file",       po::value(&optionsFile),                                       "reads the command line options from the file")
        ("network",            po::value(&m_networkFile)      ->default_value("network.jsn"), "sets the file containing the layout and weights of the neural network")
        ("cuda",               po::value(&m_useCuda)          ->default_value(true),          "use CUDA to accelerate the computations")
		("gpu_deviceid",               po::value(&m_gpu_deviceid)          ->default_value(-1),          "decide which GPU device to lock ")
        ("parallel_sequences", po::value(&m_parallelSequences)->default_value(1),             "sets the number of parallel calculated sequences")
        ("random_seed",        po::value(&m_randomSeed)       ->default_value(0u),            "sets the seed for the random number generator (0 = auto) of edge's weight")
		("random_seed_shuffle",        po::value(&m_randomseed_shuffle)       ->default_value(0u),            "sets the seed for the random number generator (0 = auto) of frac shuffle")
        ;

    po::options_description feedForwardOptions("Feed forward options");
    feedForwardOptions.add_options()
        ("ff_output_file", po::value(&m_feedForwardOutputFile)->default_value("ff_output.csv"), "sets the name of the output file in feed forward mode")
        ("ff_input_file",  po::value(&m_feedForwardInputFile),                                  "sets the name of the input file in feed forward mode")
        ;

    po::options_description trainingOptions("Training options");
    trainingOptions.add_options()
        ("train",               po::value(&m_trainingMode)     ->default_value(false),                 "enables the training mode")
        ("hybrid_online_batch", po::value(&m_hybridOnlineBatch)->default_value(false),                 "enables weight updates after every fraction of parallel calculated sequences")
        ("shuffle_fractions",   po::value(&m_shuffleFractions) ->default_value(false),                 "shuffles fractions in hybrid online/batch learning")
        ("shuffle_sequences",   po::value(&m_shuffleSequences) ->default_value(false),                 "shuffles sequences within and across fractions in hybrid online/batch learning")
        ("max_epochs",          po::value(&m_maxEpochs)        ->default_value(DEFAULT_UINT_MAX),      "sets the maximum number of training epochs")
        ("max_epochs_no_best",  po::value(&m_maxEpochsNoBest)  ->default_value(20),                    "sets the maximum number of training epochs in which no new lowest error could be achieved")
        ("validate_every",      po::value(&m_validateEvery)    ->default_value(1),                     "sets the number of epochs until the validation error is computed")
        ("test_every",          po::value(&m_testEvery)        ->default_value(1),                     "sets the number of epochs until the test error is computed")
        ("optimizer",           po::value(&optimizerString)    ->default_value("steepest_descent"),    "sets the optimizer used for updating the weights")
        ("learning_rate",       po::value(&m_learningRate)     ->default_value((real_t)1e-5, "1e-5"),  "sets the learning rate for the steepest descent optimizer")
        ("momentum",            po::value(&m_momentum)         ->default_value((real_t)0.9,  "0.9"),   "sets the momentum for the steepest descent optimizer")
        ("save_network",        po::value(&m_trainedNetwork)   ->default_value("trained_network.jsn"), "sets the file name of the trained network that will be produced")
		("save_weweights",        po::value(&s_save_weweightsFile)   ->default_value("trained_network.jsn.we"), "sets the file name of the we_weights that will be saved")
		("load_weweights",        po::value(&s_load_weweightsFile)   ->default_value("none"), "sets the file name of the we_weights that will be loaded")
		("save_featweights",        po::value(&s_save_featweightsFile)   ->default_value("trained_network.jsn.featweights"), "sets the file name of the feat_weights that will be saved")
		("load_featweights",        po::value(&s_load_featweightsFile)   ->default_value("none"), "sets the file name of the feat_weights that will be loaded")
		("load_lstmweights",        po::value(&s_load_lstmweightsFile)   ->default_value("none"), "sets the file name of the lstmLayer_weights that will be loaded")
		("cache_dir",        po::value(&m_cachedir), "sets the dir of the cache file")
		("vocab_size",            po::value(&n_vocab_size)         ->default_value(0),   "sets the vocabulary size")
		("inputwe_dim",            po::value(&n_we_dim)         ->default_value(0),   "sets the word embedding size")
		("inputfeat_dim",            po::value(&n_inputfeat_dim)         ->default_value(0),   "sets the input feat size")
		("sort_traindata",               po::value(&m_sortTrainData)          ->default_value(true),          "if sort the train data")
		("hasmultidata",               po::value(&b_hasmultidata)     ->default_value(false),                 "if has multiple data")
        ;

    po::options_description autosaveOptions("Autosave options");
    autosaveOptions.add_options()
        ("autosave",        po::value(&m_autosave)->default_value(false), "enables autosave after every epoch")
        ("autosave_prefix", po::value(&m_autosavePrefix),                 "prefix for autosave files; e.g. 'abc/mynet-' will lead to file names like 'mynet-epoch005.autosave' in the directory 'abc'")
        ("continue",        po::value(&m_continueFile),                   "continues training from an autosave file")
		("continue_cfgsrc",          po::value(&m_continue_cfgsrc)        ->default_value(0),               "sets the src of config, 0 from continue model(default), 1 from user option")
        ;

    po::options_description dataFilesOptions("Data file options");
    dataFilesOptions.add_options()
        ("train_file",        po::value(&m_trainingFile),                                 "sets the *.nc file containing the training sequences")
        ("val_file",          po::value(&m_validationFile),                               "sets the *.nc file containing the validation sequences")
        ("test_file",         po::value(&m_testFile),                                     "sets the *.nc file containing the test sequences")
        ("train_fraction",    po::value(&m_trainingFraction)  ->default_value((real_t)1), "sets the fraction of the training set to use")
        ("val_fraction",      po::value(&m_validationFraction)->default_value((real_t)1), "sets the fraction of the validation set to use")
        ("test_fraction",     po::value(&m_testFraction)      ->default_value((real_t)1), "sets the fraction of the test set to use")
        ("input_noise_sigma", po::value(&m_inputNoiseSigma)   ->default_value((real_t)0), "sets the standard deviation of the input noise for training and feed forward sets")
        ;

    po::options_description weightsInitializationOptions("Weight initialization options");
    weightsInitializationOptions.add_options()
        ("weights_dist",         po::value(&weightsDistString)   ->default_value("uniform"),            "sets the distribution type of the initial weights (uniform or normal)")
        ("weights_uniform_min",  po::value(&m_weightsUniformMin) ->default_value((real_t)-0.1, "-0.1"), "sets the minimum value of the uniform distribution")
        ("weights_uniform_max",  po::value(&m_weightsUniformMax) ->default_value((real_t)+0.1, "0.1"),  "sets the maximum value of the uniform distribution")
        ("weights_normal_sigma", po::value(&m_weightsNormalSigma)->default_value((real_t)0.1, "0.1"),   "sets the standard deviation of the normal distribution")
        ("weights_normal_mean",  po::value(&m_weightsNormalMean) ->default_value((real_t)0.0, "0"),     "sets the mean of the normal distribution")
        ;

    po::positional_options_description positionalOptions;
    positionalOptions.add("options_file", 1);

    // parse the command line
    po::options_description visibleOptions;
    visibleOptions.add(commonOptions);
    visibleOptions.add(feedForwardOptions);
    visibleOptions.add(trainingOptions);
    visibleOptions.add(autosaveOptions);
    visibleOptions.add(dataFilesOptions);
    visibleOptions.add(weightsInitializationOptions);

    po::options_description allOptions;
    allOptions.add(visibleOptions);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(allOptions).positional(positionalOptions).run(), vm);
        if (vm.count("options_file")) {
            optionsFile = vm["options_file"].as<std::string>();
            std::ifstream file(optionsFile.c_str(), std::ifstream::in);
            if (!file.is_open())
                throw std::runtime_error(std::string("Could not open options file '") + optionsFile + "'");
            po::store(po::parse_config_file(file, allOptions), vm);
        }
        po::notify(vm);
    }
    catch (const std::exception &e) {
        if (!vm.count("help"))
            std::cout << "Error while parsing the command line and/or options file: " << e.what() << std::endl;

        std::cout << "Usage: currennt [options] [options-file]" << std::endl;
        std::cout << visibleOptions;

        exit(vm.count("help") ? 0 : 1);
    }

    if (vm.count("help")) {
        std::cout << "Usage: currennt [options] [options-file]" << std::endl;
        std::cout << visibleOptions;

        exit(0);
    }

    // load options from autosave
    if (!m_continueFile.empty() && m_continue_cfgsrc==0) {
        try {
            std::stringstream ss;
            internal::deserializeOptions(m_continueFile, &ss);
            vm = po::variables_map();
            po::store(po::parse_config_file(ss, allOptions), vm);
            po::notify(vm);
        }
        catch (const std::exception &e) {
            std::cout << "Error while restoring configuration from autosave file: " << e.what() << std::endl;
            exit(1);
        }
    }

    // store the options for autosave
    m_serializedOptions = internal::serializeOptions(vm);

    // check the optimizer string
    if (optimizerString == "rprop")
        m_optimizer = OPTIMIZER_RPROP;
    else if (optimizerString == "steepest_descent")
        m_optimizer = OPTIMIZER_STEEPESTDESCENT;
    else {
        std::cout << "ERROR: Invalid optimizer. Possible values: steepest_descent, rprop." << std::endl;
        exit(1);
    }

    // create a random seed
    if (!m_randomSeed)
	{
        m_randomSeed = boost::random::random_device()();
	}
	if (!m_randomseed_shuffle)
	{
		m_randomseed_shuffle = boost::random::random_device()();
	}

    // check the weights distribution string
    if (weightsDistString == "normal")
        m_weightsDistribution = DISTRIBUTION_NORMAL;
    else if (weightsDistString == "uniform")
        m_weightsDistribution = DISTRIBUTION_UNIFORM;
    else {
        std::cout << "ERROR: Invalid initial weights distribution type. Possible values: normal, uniform." << std::endl;
        exit(1);
    }

    // check data sets fractions
    if (m_trainingFraction <= 0 || 1 < m_trainingFraction) {
        std::cout << "ERROR: Invalid training set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }
    if (m_validationFraction <= 0 || 1 < m_validationFraction) {
        std::cout << "ERROR: Invalid validation set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }
    if (m_testFraction <= 0 || 1 < m_testFraction) {
        std::cout << "ERROR: Invalid test set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }

    // print information about active command line options
    if (m_trainingMode) {
        std::cout << "Started in " << (m_hybridOnlineBatch ? "hybrid online/batch" : "batch") << " training mode." << std::endl;

        if (m_shuffleFractions)
            std::cout << "Data fractions (" << m_parallelSequences << " sequences each) will be shuffled during training." << std::endl;
        if (m_shuffleSequences)
            std::cout << "Sequences will be shuffled within and across data fractions during training." << std::endl;
        if (m_inputNoiseSigma != (real_t)0)
            std::cout << "Using input noise with a standard deviation of " << m_inputNoiseSigma << "." << std::endl;

        std::cout << "The trained network will be written to '" << m_trainedNetwork << "'." << std::endl;
        if (boost::filesystem::exists(m_trainedNetwork))
            std::cout << "WARNING: The output file '" << m_trainedNetwork << "' already exists. It will be overwritten!" << std::endl;
    }
    else {
        std::cout << "Started in feed forward mode." << std::endl;

        std::cout << "The feed forward output will be written to '" << m_feedForwardOutputFile << "'." << std::endl;
        if (boost::filesystem::exists(m_feedForwardOutputFile))
            std::cout << "WARNING: The output file '" << m_feedForwardOutputFile << "' already exists. It will be overwritten!" << std::endl;
    }

    if (m_trainingMode && !m_validationFile.empty())
        std::cout << "Validation error will be calculated every " << m_validateEvery << " epochs." << std::endl;
    if (m_trainingMode && !m_testFile.empty())
        std::cout << "Test error will be calculated every " << m_testEvery << " epochs." << std::endl;

    if (m_trainingMode) {
        std::cout << "Training will be stopped";
        if (m_maxEpochs != std::numeric_limits<unsigned>::max())
            std::cout << " after " << m_maxEpochs << " epochs or";
        std::cout << " if there is no new lowest validation error within " << m_maxEpochsNoBest << " epochs." << std::endl;
    }

    if (m_useCuda)
        std::cout << "Utilizing the GPU for computations with " << m_parallelSequences << " sequences in parallel." << std::endl;
    else
        std::cout << "WARNING: CUDA option not set. Computations will be performed on the CPU!" << std::endl;

    if (m_trainingMode) {
        if (m_weightsDistribution == DISTRIBUTION_NORMAL)
            std::cout << "Normal distribution with mean=" << m_weightsNormalMean << " and sigma=" << m_weightsNormalSigma;
        else
            std::cout << "Uniform distribution with range [" << m_weightsUniformMin << ", " << m_weightsUniformMax << "]";
        std::cout << ". Random seed: " << m_randomSeed <<" Random seed shuffle: "<<m_randomseed_shuffle <<std::endl;
    }

	// lock gpu
	if (m_useCuda) {
		if(m_gpu_deviceid < 0) {
			int total_gpudevicenum = 0;
			cudaGetDeviceCount(&total_gpudevicenum);
			m_gpu_deviceid = 0; 
			while(m_gpu_deviceid < total_gpudevicenum) {
				if(lockgpudevice(m_gpu_deviceid)) {
					cudaSetDevice(m_gpu_deviceid);
					break;
				}
				++m_gpu_deviceid;
			}
			if(m_gpu_deviceid == total_gpudevicenum)
				exit(0);
		}
		else {// force to use
			if(lockgpudevice(m_gpu_deviceid))
				cudaSetDevice(m_gpu_deviceid);
			else
				std::cerr<<"Warning: can not lock gpu device, just run on gpu: "<<m_gpu_deviceid<<std::endl;
				cudaSetDevice(m_gpu_deviceid);
		}
	}


    std::cout << std::endl;
}

Configuration::~Configuration()
{
}

bool Configuration::lockgpudevice(int device_id)
{
    wchar_t buffer[80];
    wsprintfW (buffer, L"Global\\DBN.exe GPGPU exclusive lock for device %d", device_id);
	fprintf (stderr, "->currennt.exe is trying to lock GPU device %d\n", device_id);
    // we actually use a Windows-wide named mutex
    HANDLE h = ::CreateMutexW (NULL/*security attr*/, TRUE/*bInitialOwner*/,buffer);
    DWORD res = ::GetLastError();
    if (h == NULL)  // failure  --this should not really happen
    {
        if (res == ERROR_ACCESS_DENIED)    // no access: already locked by another process
        {
            fprintf (stderr, "->cannot lock device: mutex access denied, assuming already locked mutex '%d'\n", device_id);
            return false;
        }
        fprintf (stderr, "->cannot lock device: failed to create mutex '%d': error %d\n", device_id, res);
        throw std::runtime_error ("->cannot lock device: unexpected failure\n");
    }
    // got a handle
    if (res == 0)   // no error
    {
        fprintf (stderr, "->lock device successfully: locked mutex '%d'\n", device_id);
        return true;
    }
    // failure with handle  --remember to release the handle
    ::CloseHandle (h);
    if (res == ERROR_ALREADY_EXISTS)    // already locked by another process
    {
        fprintf (stderr, "->cannot lock device: mutex '%d' is already locked\n", device_id);
        return false;
    }
    else if (res != 0)
    {
        fprintf (stderr, "->cannot lock device: unexpected error from CreateMutex() when attempting to create and acquire mutex '%S': %d\n", buffer, res);
        throw std::logic_error ("->cannot lock device: unexpected failure\n");
    }
    return false;
}

const Configuration& Configuration::instance()
{
    return *ms_instance;
}

const std::string& Configuration::serializedOptions() const
{
    return m_serializedOptions;
}

bool Configuration::trainingMode() const
{
    return m_trainingMode;
}

bool Configuration::hybridOnlineBatch() const
{
    return m_hybridOnlineBatch;
}

bool Configuration::shuffleFractions() const
{
    return m_shuffleFractions;
}

bool Configuration::shuffleSequences() const
{
    return m_shuffleSequences;
}

bool Configuration::useCuda() const
{
    return m_useCuda;
}

bool Configuration::autosave() const
{
    return m_autosave;
}

Configuration::optimizer_type_t Configuration::optimizer() const
{
    return m_optimizer;
}

int Configuration::parallelSequences() const
{
    return (int)m_parallelSequences;
}

int Configuration::maxEpochs() const
{
    return (int)m_maxEpochs;
}

int Configuration::maxEpochsNoBest() const
{
    return (int)m_maxEpochsNoBest;
}

int Configuration::validateEvery() const
{
    return (int)m_validateEvery;
}

int Configuration::testEvery() const
{
    return (int)m_testEvery;
}

real_t Configuration::learningRate() const
{
    return m_learningRate;
}

real_t Configuration::momentum() const
{
    return m_momentum;
}

const std::string& Configuration::networkFile() const
{
    return m_networkFile;
}

const std::string& Configuration::trainingFile() const
{
    return m_trainingFile;
}

const std::string& Configuration::validationFile() const
{
    return m_validationFile;
}

const std::string& Configuration::testFile() const
{
    return m_testFile;
}

unsigned Configuration::randomSeed() const
{
    return m_randomSeed;
}

Configuration::distribution_type_t Configuration::weightsDistributionType() const
{
    return m_weightsDistribution;
}

real_t Configuration::weightsDistributionUniformMin() const
{
    return m_weightsUniformMin;
}

real_t Configuration::weightsDistributionUniformMax() const
{
    return m_weightsUniformMax;
}

real_t Configuration::weightsDistributionNormalSigma() const
{
    return m_weightsNormalSigma;
}

real_t Configuration::weightsDistributionNormalMean() const
{
    return m_weightsNormalMean;
}

real_t Configuration::inputNoiseSigma() const
{
    return m_inputNoiseSigma;
}

real_t Configuration::trainingFraction() const
{
    return m_trainingFraction;
}

real_t Configuration::validationFraction() const
{
    return m_validationFraction;
}

real_t Configuration::testFraction() const
{
    return m_testFraction;
}

const std::string& Configuration::trainedNetworkFile() const
{
    return m_trainedNetwork;
}

const std::string& Configuration::feedForwardInputFile() const
{
    return m_feedForwardInputFile;

}
const std::string& Configuration::feedForwardOutputFile() const
{
    return m_feedForwardOutputFile;
}

const std::string& Configuration::autosavePrefix() const
{
    return m_autosavePrefix;
}

const std::string& Configuration::continueFile() const
{
    return m_continueFile;
}

const std::string& Configuration::cacheDir() const
{
    return m_cachedir;
}

unsigned Configuration::randomSeedShuffle() const
{
    return m_randomseed_shuffle;
}

int Configuration::vocabSize() const
{
    return n_vocab_size;
}

int Configuration::inputWeDim() const
{
    return n_we_dim;
}

int Configuration::inputFeatDim() const
{
    return n_inputfeat_dim;
}

bool Configuration::sortTrainData() const
{
    return m_sortTrainData;
}

const std::string& Configuration::saveWeweightsFile() const
{
    return s_save_weweightsFile;
}

const std::string& Configuration::loadWeweightsFile() const
{
    return s_load_weweightsFile;
}

const std::string& Configuration::saveFeatweightsFile() const
{
    return s_save_featweightsFile;
}

const std::string& Configuration::loadFeatweightsFile() const
{
    return s_load_featweightsFile;
}

const std::string& Configuration::loadLstmweightsFile() const
{
    return s_load_lstmweightsFile;
}

bool Configuration::hasMultiData() const
{
    return b_hasmultidata;
}
