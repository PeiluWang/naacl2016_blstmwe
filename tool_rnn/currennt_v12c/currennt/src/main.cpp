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

#include "../../currennt_lib/src/Configuration.hpp"
#include "../../currennt_lib/src/NeuralNetwork.hpp"
#include "../../currennt_lib/src/layers/LstmLayer.hpp"
#include "../../currennt_lib/src/layers/BinaryClassificationLayer.hpp"
#include "../../currennt_lib/src/layers/MulticlassClassificationLayer.hpp"
#include "../../currennt_lib/src/optimizers/SteepestDescentOptimizer.hpp"
#include "../../currennt_lib/src/helpers/JsonClasses.hpp"
#include "../../currennt_lib/src/rapidjson/prettywriter.h"
#include "../../currennt_lib/src/rapidjson/filestream.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <stdarg.h>


enum data_set_type
{
    DATA_SET_TRAINING,
    DATA_SET_VALIDATION,
    DATA_SET_TEST,
    DATA_SET_FEEDFORWARD
};

// helper functions (implementation below)
void readJsonFile(rapidjson::Document *doc, const std::string &filename);
boost::shared_ptr<data_sets::DataSet> loadDataSet(data_set_type dsType);
template <typename TDevice> void printLayers(const NeuralNetwork<TDevice> &nn);
template <typename TDevice> void printOptimizer(const optimizers::Optimizer<TDevice> &optimizer);
template <typename TDevice> void saveNetwork(const NeuralNetwork<TDevice> &nn, const std::string &filename);
void createModifiedTrainingSet(data_sets::DataSet *trainingSet, int parallelSequences, bool outputsToClasses, boost::mutex &swapTrainingSetsMutex);
template <typename TDevice> void saveState(const NeuralNetwork<TDevice> &nn, const optimizers::Optimizer<TDevice> &optimizer, const std::string &infoRows);
template <typename TDevice> void restoreState(NeuralNetwork<TDevice> *nn, optimizers::Optimizer<TDevice> *optimizer, std::string *infoRows);
std::string printfRow(const char *format, ...);


// main function
template <typename TDevice>
int trainerMain(const Configuration &config)
{
    try {
        // read the neural network description file 
        std::string networkFile = config.continueFile().empty() ? config.networkFile() : config.continueFile();
        printf("Reading network from '%s'... ", networkFile.c_str());
        fflush(stdout);
        rapidjson::Document netDoc;
        readJsonFile(&netDoc, networkFile);
        printf("done.\n");
        printf("\n");

        // load data sets
        boost::shared_ptr<data_sets::DataSet> trainingSet    = boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> validationSet  = boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> testSet        = boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> feedForwardSet = boost::make_shared<data_sets::DataSet>();

		int inputFeatDim=0;
		int outputDim=0;

        if (config.trainingMode()) {
            trainingSet = loadDataSet(DATA_SET_TRAINING);
			
            if (!config.validationFile().empty())
                validationSet = loadDataSet(DATA_SET_VALIDATION);
            
            if (!config.testFile().empty())
                testSet = loadDataSet(DATA_SET_TEST);

			inputFeatDim=trainingSet->inputFeatDim();
			outputDim=trainingSet->outputDim();
        }
        else {
            feedForwardSet = loadDataSet(DATA_SET_FEEDFORWARD);

			inputFeatDim=feedForwardSet->inputFeatDim();
			outputDim=feedForwardSet->outputDim();
        }

        // calculate the maximum sequence length
        int maxSeqLength;
        if (config.trainingMode())
            maxSeqLength = std::max(trainingSet->maxSeqLength(), std::max(validationSet->maxSeqLength(), testSet->maxSeqLength()));
        else
            maxSeqLength = feedForwardSet->maxSeqLength();

        int parallelSequences = config.parallelSequences();
        
        // create the neural network
        printf("Creating the neural network... ");
        fflush(stdout);
        NeuralNetwork<TDevice> neuralNetwork(netDoc, parallelSequences, maxSeqLength);
        printf("done.\n");
        printf("Layers:\n");
        printLayers(neuralNetwork);
        printf("\n");

		// check layer size consistency
		int inputlayer_size=neuralNetwork.inputLayer().size();
		int outputlayer_size=neuralNetwork.outputLayer().size();
		printf("\ninput feat dim: %d\n", config.inputFeatDim());
		printf("input we dim: %d\n", config.inputWeDim());
		printf("\nvocab size: %d\n", config.vocabSize());
		assert(inputlayer_size == config.inputWeDim());
		assert(outputlayer_size == outputDim);
		assert(config.inputFeatDim()==inputFeatDim);

        // check if this is a classification task
        bool classificationTask = false;
        if (dynamic_cast<layers::BinaryClassificationLayer<TDevice>*>(&neuralNetwork.postOutputLayer())
            || dynamic_cast<layers::MulticlassClassificationLayer<TDevice>*>(&neuralNetwork.postOutputLayer())) {
                classificationTask = true;
        }

        printf("\n");

        // create the optimizer
        if (config.trainingMode()) {
            printf("Creating the optimizer... ");
            fflush(stdout);
            boost::scoped_ptr<optimizers::Optimizer<TDevice> > optimizer;

            switch (config.optimizer()) {
            case Configuration::OPTIMIZER_STEEPESTDESCENT:
                optimizer.reset(new optimizers::SteepestDescentOptimizer<TDevice>(
                    neuralNetwork, *trainingSet, *validationSet, *testSet,
                    config.maxEpochs(), config.maxEpochsNoBest(), config.validateEvery(), config.testEvery(),
                    config.learningRate(), config.momentum()
                    ));
                break;

            default:
                throw std::runtime_error("Unknown optimizer type");
            }

            printf("done.\n");
            printOptimizer(config, *optimizer);

            std::string infoRows;

            // continue from autosave?
            if (!config.continueFile().empty()) {
                printf("Restoring state from '%s'... ", config.continueFile().c_str());
                fflush(stdout);
                restoreState(&neuralNetwork, &*optimizer, &infoRows);
                printf("done.\n\n");
            }

            // train the network
            printf("Starting training...\n");
            printf("\n");

            printf(" Epoch | Duration |  Training error  | Validation error |    Test error    | New best \n");
            printf("-------+----------+------------------+------------------+------------------+----------\n");
            std::cout << infoRows;

            bool finished = false;
            while (!finished) {
                const char *errFormat = (classificationTask ? "%6.2lf%%%10.3lf |" : "%17.3lf |");
                const char *errSpace  = "                  |";

                // train for one epoch and measure the time
                infoRows += printfRow(" %5d | ", optimizer->currentEpoch() + 1);
                
                boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::local_time();
                finished = optimizer->train();
                boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
                double duration = (double)(endTime - startTime).total_milliseconds() / 1000.0;

                infoRows += printfRow("%8.1lf |", duration);
                if (classificationTask)
                    infoRows += printfRow(errFormat, (double)optimizer->curTrainingClassError()*100.0, (double)optimizer->curTrainingError());
                else
                    infoRows += printfRow(errFormat, (double)optimizer->curTrainingError());
                
                if (!validationSet->empty() && optimizer->currentEpoch() % config.validateEvery() == 0) {
                    if (classificationTask)
                        infoRows += printfRow(errFormat, (double)optimizer->curValidationClassError()*100.0, (double)optimizer->curValidationError());
                    else
                        infoRows += printfRow(errFormat, (double)optimizer->curValidationError());
                }
                else
                    infoRows += printfRow("%s", errSpace);

                if (!testSet->empty() && optimizer->currentEpoch() % config.testEvery() == 0) {
                    if (classificationTask)
                        infoRows += printfRow(errFormat, (double)optimizer->curTestClassError()*100.0, (double)optimizer->curTestError());
                    else
                        infoRows += printfRow(errFormat, (double)optimizer->curTestError());
                }
                else
                    infoRows += printfRow("%s", errSpace);

                if (!validationSet->empty() && optimizer->currentEpoch() % config.validateEvery() == 0) {
                    if (optimizer->epochsSinceLowestValidationError() == 0)
                        infoRows += printfRow("  yes   \n");
                    else
                        infoRows += printfRow("  no    \n");
                }
                else
                    infoRows += printfRow("        \n");

                // autosave
                if (config.autosave())
                    saveState(neuralNetwork, *optimizer, infoRows);
            }

            printf("\n");

            if (optimizer->epochsSinceLowestValidationError() == config.maxEpochsNoBest())
                printf("No new lowest error since %d epochs. Training stopped.\n", config.maxEpochsNoBest());
            else
                printf("Maximum number of training epochs reached. Training stopped.\n");

            printf("Lowest validation error: %lf\n", optimizer->lowestValidationError());
            printf("\n");

            // save the trained network to the output file
            saveNetwork(neuralNetwork, config.trainedNetworkFile());
        }
        // evaluation mode
        else {
            // open the output file
            std::ofstream file(config.feedForwardOutputFile().c_str(), std::ofstream::out);

            // process all data set fractions
            int fracIdx = 0;
            boost::shared_ptr<data_sets::DataSetFraction> frac;
            while (((frac = feedForwardSet->getNextFraction()))) {
                printf("Computing outputs for data fraction %d...", ++fracIdx);
                fflush(stdout);

                // compute the forward pass for the current data fraction and extract the outputs
                neuralNetwork.loadSequences(*frac);
                neuralNetwork.computeForwardPass();
                std::vector<std::vector<std::vector<real_t> > > outputs = neuralNetwork.getOutputs();

                // write the outputs in the file
                for (int psIdx = 0; psIdx < (int)outputs.size(); ++psIdx) {
                    // write the sequence tag
                    file << frac->seqInfo(psIdx).seqTag;

                    // write the patterns
                    for (int timestep = 0; timestep < (int)outputs[psIdx].size(); ++timestep) {
                        for (int outputIdx = 0; outputIdx < (int)outputs[psIdx][timestep].size(); ++outputIdx)
                            file << ';' << outputs[psIdx][timestep][outputIdx];
                    }

                    file << '\n';
                }

                printf(" done.\n");
            }

            // close the file
            file.close();
        }
    }
    catch (const std::exception &e) {
        printf("FAILED: %s\n", e.what());
        return 2;
    }

    return 0;
}


int main(int argc, const char *argv[])
{
    // load the configuration
    Configuration config(argc, argv);

    // run the execution device specific main function
    if (config.useCuda())
        return trainerMain<Gpu>(config);
    else
        return trainerMain<Cpu>(config);
}


void readJsonFile(rapidjson::Document *doc, const std::string &filename)
{
    // open the file
    std::ifstream ifs(filename.c_str(), std::ios::binary);
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

    std::string docStr(buffer);
    delete buffer;

    // extract the JSON tree
    if (doc->Parse<0>(docStr.c_str()).HasParseError())
        throw std::runtime_error(std::string("Parse error: ") + doc->GetParseError());
}


boost::shared_ptr<data_sets::DataSet> loadDataSet(data_set_type dsType)
{
    std::string type;
    std::string filename;
    real_t fraction = 1;
    bool fracShuf   = false;
    bool seqShuf    = false;
    real_t noiseDev = 0;

    switch (dsType) {
    case DATA_SET_TRAINING:
        type     = "training set";
        filename = Configuration::instance().trainingFile();
        fraction = Configuration::instance().trainingFraction();
        fracShuf = Configuration::instance().shuffleFractions();
        seqShuf  = Configuration::instance().shuffleSequences();
        noiseDev = Configuration::instance().inputNoiseSigma();
        break;

    case DATA_SET_VALIDATION:
        type     = "validation set";
        filename = Configuration::instance().validationFile();
        fraction = Configuration::instance().validationFraction();
        break;

    case DATA_SET_TEST:
        type     = "test set";
        filename = Configuration::instance().testFile();
        fraction = Configuration::instance().testFraction();
        break;

    default:
        type     = "feed forward input set";
        filename = Configuration::instance().feedForwardInputFile();
        noiseDev = Configuration::instance().inputNoiseSigma();
        break;
    }

    printf("Loading %s '%s'... ", type.c_str(), filename.c_str());
    fflush(stdout);

    boost::shared_ptr<data_sets::DataSet> ds = boost::make_shared<data_sets::DataSet>(
        filename, Configuration::instance().parallelSequences(), fraction, fracShuf, seqShuf, noiseDev);

    printf("done.\n");
    printf("Loaded fraction:  %d%%\n",   (int)(fraction*100));
    printf("Sequences:        %d\n",     ds->totalSequences());
    printf("Sequence lengths: %d..%d\n", ds->minSeqLength(), ds->maxSeqLength());
    printf("Total timesteps:  %d\n",     ds->totalTimesteps());
    printf("\n");

    return ds;
}


template <typename TDevice>
void printLayers(const NeuralNetwork<TDevice> &nn)
{
    int weights = 0;

    for (int i = 0; i < (int)nn.layers().size(); ++i) {
        printf("(%d) %s ", i, nn.layers()[i]->type().c_str());
        printf("[size: %d", nn.layers()[i]->size());

        const layers::TrainableLayer<TDevice>* tl = dynamic_cast<const layers::TrainableLayer<TDevice>*>(nn.layers()[i].get());
        if (tl) {
            printf(", bias: %.1lf, weights: %d", (double)tl->bias(), (int)tl->weights().size());
            weights += (int)tl->weights().size();
        }

        printf("]\n");
    }

    printf("Total weights: %d\n", weights);
}


template <typename TDevice> 
void printOptimizer(const Configuration &config, const optimizers::Optimizer<TDevice> &optimizer)
{
    if (dynamic_cast<const optimizers::SteepestDescentOptimizer<TDevice>*>(&optimizer)) {
        printf("Optimizer type: Steepest descent with momentum\n");
        printf("Max training epochs:       %d\n", config.maxEpochs());
        printf("Max epochs until new best: %d\n", config.maxEpochsNoBest());
        printf("Validation error every:    %d\n", config.validateEvery());
        printf("Test error every:          %d\n", config.testEvery());
        printf("Learning rate:             %g\n", (double)config.learningRate());
        printf("Momentum:                  %g\n", (double)config.momentum());
        printf("\n");
    }
}


template <typename TDevice> 
void saveNetwork(const NeuralNetwork<TDevice> &nn, const std::string &filename)
{
    printf("Storing the trained network in '%s'... ", filename.c_str());

    rapidjson::Document jsonDoc;
    jsonDoc.SetObject();
    nn.exportLayers (&jsonDoc);
    nn.exportWeights(&jsonDoc);
	nn.exportWordEmbedding(Configuration::instance().saveWeweightsFile());
	nn.exportFeatWeights(Configuration::instance().saveFeatweightsFile());

    FILE *file = fopen(filename.c_str(), "w");
    if (!file)
        throw std::runtime_error("Cannot open file");

    rapidjson::FileStream os(file);
    rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
    jsonDoc.Accept(writer);

    fclose(file);

    printf("done.\n");
    printf("\n");
}


template <typename TDevice> 
void saveState(const NeuralNetwork<TDevice> &nn, const optimizers::Optimizer<TDevice> &optimizer, const std::string &infoRows)
{
    // create the JSON document
    rapidjson::Document jsonDoc;
    jsonDoc.SetObject();

    // add the configuration options
    jsonDoc.AddMember("configuration", Configuration::instance().serializedOptions().c_str(), jsonDoc.GetAllocator());

    // add the info rows
    std::string tmp = boost::replace_all_copy(infoRows, "\n", ";;;");
    jsonDoc.AddMember("info_rows", tmp.c_str(), jsonDoc.GetAllocator());

    // add the network structure and weights
    nn.exportLayers (&jsonDoc);
    nn.exportWeights(&jsonDoc);

    // add the state of the optimizer
    optimizer.exportState(&jsonDoc);
    
    // open the file
    char buffer[100];
    sprintf(buffer, "%sepoch%03d.autosave", Configuration::instance().autosavePrefix().c_str(), optimizer.currentEpoch());
    FILE *file = fopen(buffer, "w");
    if (!file)
        throw std::runtime_error("Cannot open file");

    // write the file
    rapidjson::FileStream os(file);
    rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
    jsonDoc.Accept(writer);
    fclose(file);

	// save the word embedding
	sprintf(buffer, "%sepoch%03d.autosave.we", Configuration::instance().autosavePrefix().c_str(), optimizer.currentEpoch());
	std::string we_autosave_file=buffer;
	nn.exportWordEmbedding(we_autosave_file);

	// save the feat weights
	sprintf(buffer, "%sepoch%03d.autosave.featweights", Configuration::instance().autosavePrefix().c_str(), optimizer.currentEpoch());
	std::string feat_autosave_file=buffer;
	nn.exportFeatWeights(feat_autosave_file);
}


template <typename TDevice> 
void restoreState(NeuralNetwork<TDevice> *nn, optimizers::Optimizer<TDevice> *optimizer, std::string *infoRows)
{
    rapidjson::Document jsonDoc;
    readJsonFile(&jsonDoc, Configuration::instance().continueFile());

    // extract info rows
    if (!jsonDoc.HasMember("info_rows"))
        throw std::runtime_error("Missing value 'info_rows'");
    *infoRows = jsonDoc["info_rows"].GetString();
    boost::replace_all(*infoRows, ";;;", "\n");

    // extract the state of the optimizer
    optimizer->importState(jsonDoc);
}


std::string printfRow(const char *format, ...)
{
    // write to temporary buffer
    char buffer[100];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // print on stdout
    std::cout << buffer;
    fflush(stdout);

    // return the same string
    return std::string(buffer);
}
