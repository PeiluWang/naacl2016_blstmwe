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

#include "InputLayer.hpp"
#include "../Configuration.hpp"
#include "../helpers/Matrix.hpp"
#include "../helpers/NumericLimits.cuh"
#include "../helpers/getRawPointer.cuh"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <boost/lexical_cast.hpp>
#include <stdexcept>
#include <math.h>
#include <fstream>

struct UpdateWeWeights
{
	real_t learningrate;
	real_t momentum;

	real_t* weweights;
	real_t* weupdateweights;
	real_t* lastweupdateweights;

	__host__ __device__
	real_t operator()(const int &id) const { 
		lastweupdateweights[id]=momentum*lastweupdateweights[id]-(1-momentum)*learningrate*weupdateweights[id];

		real_t newweweights=weweights[id]+lastweupdateweights[id];
		return newweweights;
	}
};

struct UpdateFeatWeights
{
	real_t learningrate;
	real_t momentum;

	real_t* featweights;
	real_t* featupdateweights;
	real_t* lastfeatupdateweights;

	__host__ __device__
	real_t operator()(const int &id) const { 
		lastfeatupdateweights[id]=momentum*lastfeatupdateweights[id]-(1-momentum)*learningrate*featupdateweights[id];

		real_t newweweights=featweights[id]+lastfeatupdateweights[id];
		return newweweights;
	}
};

struct LogisticForward
{
	__host__ __device__
	real_t operator()(const real_t& x) const { 
		real_t n_x=activation_functions::Tanh::fn(x);
		//real_t n_x=log(1.0+exp(x));
		/*
		real_t n_x=x;
		if(x<0){
			n_x=0;
		}*/
		return n_x;
	}
};

struct ComputeDelta
    {
        // since calculating the derivatives is very cheap for our activation functions, 
        // we simple calculate the deltas of all timesteps, including dummies
        
        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&> &t) const
        {
            real_t delta = activation_functions::Tanh::deriv(t.get<1>()) * t.get<0>();
			//real_t delta = 1.0/(1.0+exp(-t.get<1>()))*t.get<0>(); //rectifier
			/*
			real_t delta=t.get<0>();
			if(t.get<1>()<0){
				delta=0;
			}*/
            t.get<0>() = delta;
        }
    };

namespace layers {

    template <typename TDevice>
    InputLayer<TDevice>::InputLayer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength)
        : Layer<TDevice>(layerChild, parallelSequences, maxSeqLength)
    {
		const Configuration &config = Configuration::instance();
		int vocab_size=config.vocabSize();
		inputWeDim=config.inputWeDim();
		inputFeatDim=config.inputFeatDim();

		assert(vocab_size>0);
		assert(inputWeDim>0);
		assert(inputFeatDim>0);

		// init we_weights
		Cpu::real_vector weights(vocab_size*inputWeDim);
		Cpu::real_vector fweights(inputFeatDim*inputWeDim);

		bool loadweweights=false;
		if(config.loadWeweightsFile()!="none"){
			loadweweights=true;
		}
		if(!loadweweights){// init we weights randomly
			std::cout<<"\ninit weweights randomly\n"<<std::endl;
			static boost::mt19937 *gen = NULL;
			if (!gen) {
				gen = new boost::mt19937;
				gen->seed(config.randomSeed());
			}
			if (config.weightsDistributionType() == Configuration::DISTRIBUTION_UNIFORM) {
				real_t range = config.weightsDistributionUniformMax() - config.weightsDistributionUniformMin();
				boost::random::uniform_real_distribution<real_t> dist(0, range);
				for (size_t i = 0; i < weights.size(); ++i)
					weights[i] = dist(*gen) + config.weightsDistributionUniformMin();
			}
			else {
				boost::random::normal_distribution<real_t> dist(config.weightsDistributionNormalMean(), config.weightsDistributionNormalSigma());
				for (size_t i = 0; i < weights.size(); ++i)
					weights[i] = dist(*gen);
			}
		}else{// load we weights from file
			std::string wedict_file=config.loadWeweightsFile();
			std::cout<<"\nload weweights: "<<wedict_file<<std::endl;
			std::ifstream fin(wedict_file);
			if(!fin){
				std::cerr<<"load wedict_file exception: "<<wedict_file<<std::endl;
				throw std::runtime_error(std::string("wedict_file not exist!"));
			}
			std::string line;
			int i=0;
			int word_num=0;
			while(std::getline(fin,line)){
				int s=0;
				int e=0;
				int senlen=line.length();
				while(true){
					e=(int)line.find(" ",s);
					if(e<0){
						break;
					}
					std::string value=line.substr(s,e-s);
					float v=(float)std::atof(value.c_str());
					weights[i]=v;
					i+=1;
					s=e+1;
				}
				word_num+=1;
			}
			std::cout<<"load complete. word num:"<<word_num<<" total value:"<<i<<std::endl;
		}
		//copy weights to we_weights
		we_weights=weights;
		last_weweightUpdates=weights;

		bool loadfeatweights=false;
		if(config.loadFeatweightsFile()!="none"){
			loadfeatweights=true;
		}
		if(!loadfeatweights){// init feat weights randomly
			std::cout<<"\ninit featweights randomly\n"<<std::endl;
			static boost::mt19937 *gen = NULL;
			if (!gen) {
				gen = new boost::mt19937;
				gen->seed(config.randomSeed());
			}
			if (config.weightsDistributionType() == Configuration::DISTRIBUTION_UNIFORM) {
				real_t range = config.weightsDistributionUniformMax() - config.weightsDistributionUniformMin();
				boost::random::uniform_real_distribution<real_t> dist(0, range);
				for (size_t i = 0; i < fweights.size(); ++i)
					fweights[i] = dist(*gen) + config.weightsDistributionUniformMin();
			}
			else {
				boost::random::normal_distribution<real_t> dist(config.weightsDistributionNormalMean(), config.weightsDistributionNormalSigma());
				for (size_t i = 0; i < fweights.size(); ++i)
					fweights[i] = dist(*gen);
			}
		}else{// load feat weights from file
			std::string featweights_file=config.loadFeatweightsFile();
			std::cout<<"\nload featweights: "<<featweights_file<<std::endl;
			std::ifstream fin(featweights_file);
			if(!fin){
				std::cerr<<"load featweights exception: "<<featweights_file<<std::endl;
				throw std::runtime_error(std::string("featweights not exist!"));
			}
			std::stringstream buffer;
			buffer << fin.rdbuf();
			std::string line(buffer.str());

			int i=0;
			int s=0;
			int e=0;
			while(true){
				e=(int)line.find(" ",s);
				if(e<0){
					break;
				}
				std::string value=line.substr(s,e-s);
				float v=(float)std::atof(value.c_str());
				fweights[i]=v;
				i+=1;
				s=e+1;
			}
			
			std::cout<<"load complete. weights size:"<<i<<std::endl;
		}
		//copy weights to feat_weights
		feat_weights=fweights;
		feat_weightUpdates=fweights;

		last_featweightUpdates=fweights;
		thrust::fill(last_weweightUpdates.begin(), last_weweightUpdates.end(), 0);
		thrust::fill(last_featweightUpdates.begin(), last_featweightUpdates.end(), 0);

		v_outputErrors = Cpu::real_vector(this->_outputs().size(), (real_t)0);
    }

    template <typename TDevice>
    InputLayer<TDevice>::~InputLayer()
    {
    }

    template <typename TDevice>
    const std::string& InputLayer<TDevice>::type() const
    {
        static const std::string s("input");
        return s;
    }

    template <typename TDevice>
    void InputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        Layer<TDevice>::loadSequences(fraction);
		thrust::fill(this->_outputs().begin(),this->_outputs().end(),0);

		int parallelnum=this->parallelSequences();
		int seqnum=fraction.numSequences();
		int layersize=this->size();

		assert(layersize==inputWeDim);

		v_inputwords=fraction.inputWords();
		v_inputfeats=fraction.inputFeats();

		// load word embedding
		for(int pid = 0; pid < seqnum; ++pid){
			int seqlength=fraction.seqInfo(pid).length;
			for(int stepid = 0; stepid < seqlength; ++stepid){
				int t=stepid * parallelnum + pid;
				//load input word, applying word embedding
				int wordid=v_inputwords[t];
				assert(wordid>=0);
				thrust::copy_n(we_weights.begin()+wordid*inputWeDim,
					inputWeDim,
					this->_outputs().begin()+inputWeDim * t);
			}
		}
    }

    template <typename TDevice>
    void InputLayer<TDevice>::computeForwardPass()
    {
		{{
			// calculate inputFeat forward
			helpers::Matrix<TDevice> weightsMatrix (&feat_weights, inputFeatDim, inputWeDim);
			helpers::Matrix<TDevice> plOutputsMatrix(&v_inputfeats, inputFeatDim, this->curMaxSeqLength() * this->parallelSequences());
			helpers::Matrix<TDevice> outputsMatrix  (&this->_outputs(), inputWeDim, this->curMaxSeqLength() * this->parallelSequences());

			outputsMatrix.addProduct(weightsMatrix, true, plOutputsMatrix, false);
		}}
		/*
		{{
			// calculate logistic
			LogisticForward fn;
			thrust::transform(this->_outputs().begin(),
				this->_outputs().end(),
				this->_outputs().begin(),
				fn);
		}}
		*/
    }

    template <typename TDevice>
    void InputLayer<TDevice>::computeBackwardPass()
    {
		float n_learningrate=Configuration::instance().learningRate();
		int layersize = this->size();
		float momentum=Configuration::instance().momentum();
		/*
		// compute delta
		{{
			ComputeDelta fndelta;

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(),   this->outputs().begin())),
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n, this->outputs().begin()+n)),
                fndelta
                );
		}}
		*/
		UpdateWeWeights fn;
		fn.learningrate=n_learningrate;
		fn.momentum=momentum;
		real_t* weweights=helpers::getRawPointer(we_weights);
		real_t* outputErrors=helpers::getRawPointer(v_outputErrors);
		real_t* lastweupdateweights=helpers::getRawPointer(last_weweightUpdates);

        // update we_weights
		for(int i=0;i<v_inputwords.size();++i){
			try{
			int wordid=v_inputwords[i];
			if(wordid<0){
				continue;
			}
			int offset=wordid*inputWeDim;
			fn.weweights=weweights+offset;
			fn.weupdateweights=outputErrors+i*layersize;
			fn.lastweupdateweights=lastweupdateweights+offset;

			thrust::transform(
				thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>(inputWeDim),
				we_weights.begin()+offset,
				fn);
			}catch(...){
				int wordid=v_inputwords[i];
				std::cerr<<"update inputLayer we weights error! wordid:"<<wordid<<" i:"<<i<<std::endl;
				throw std::runtime_error(std::string("update we weights error!"));
			}
		}
		// compute the weight updates
        {{
            helpers::Matrix<TDevice> weightUpdatesMatrix(&feat_weightUpdates, inputFeatDim, inputWeDim);
            helpers::Matrix<TDevice> plOutputsMatrix (&v_inputfeats, inputFeatDim, this->curMaxSeqLength() * this->parallelSequences());
            helpers::Matrix<TDevice> deltasMatrix (&this->outputErrors(), inputWeDim, this->curMaxSeqLength() * this->parallelSequences());

            weightUpdatesMatrix.assignProduct(plOutputsMatrix, false, deltasMatrix, true);
        }}
		// update feat_weights
		{{
			UpdateFeatWeights fn2;
			fn2.learningrate=n_learningrate;
			fn2.momentum=momentum;
			fn2.featweights=helpers::getRawPointer(feat_weights);
			fn2.featupdateweights=helpers::getRawPointer(feat_weightUpdates);
			fn2.lastfeatupdateweights=helpers::getRawPointer(last_featweightUpdates);
			
			thrust::transform(
				thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>((int)feat_weights.size()),
				feat_weights.begin(),
				fn2);
		}}
    }

	template <typename TDevice>
    typename InputLayer<TDevice>::real_vector& InputLayer<TDevice>::outputErrors()
    {
        return v_outputErrors;
    }

	template <typename TDevice>
    typename InputLayer<TDevice>::real_vector& InputLayer<TDevice>::weWeights()
    {
        return we_weights;
    }
	
	template <typename TDevice>
    typename InputLayer<TDevice>::real_vector& InputLayer<TDevice>::featWeights()
    {
        return feat_weights;
    }

    // explicit template instantiations
    template class InputLayer<Cpu>;
    template class InputLayer<Gpu>;

} // namespace layers
