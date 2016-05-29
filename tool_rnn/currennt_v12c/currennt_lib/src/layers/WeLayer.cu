
#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "WeLayer.hpp"
#include "../Configuration.hpp"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>

#include <stdexcept>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <typeinfo>

#include <fstream>


struct saxpy_functor
{
	const real_t a;

	saxpy_functor(real_t _a) : a(_a) {}

	__host__ __device__
		real_t operator()(const real_t& x, const real_t& y) const { 
			return x - a*y;
		}
};


namespace layers {

	template <typename TDevice>
    WeLayer<TDevice>::WeLayer(
        const helpers::JsonValue &layerChild, 
        const helpers::JsonValue &weightsSection,
        Layer<TDevice> &precedingLayer)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 1, 0, precedingLayer)
    {
		
		const Configuration &config = Configuration::instance();
		vocab_size=config.vocabSize();
		if(vocab_size<=0){
			throw std::runtime_error(std::string("vocab_size<=0!"));
		}
		int we_dim=this->size();
		//init we_weights
		Cpu::real_vector weights(vocab_size*we_dim);

		bool trainingmode=config.trainingMode();
		if(trainingmode){// init we weights randomly
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
		}else{// load from file
			std::string wedict_file=config.networkFile()+".we";
			std::cout<<"\nload wedict: "<<wedict_file<<std::endl;
			std::ifstream fin(wedict_file);
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

		n_learningrate=config.learningRate();
		
    }

    template <typename TDevice>
    WeLayer<TDevice>::~WeLayer()
    {
    }

	template <typename TDevice>
    const std::string& WeLayer<TDevice>::type() const
    {
        static std::string s="welayer";
        return s;
		
    }
	
	template <typename TDevice>
    void WeLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
		Layer<TDevice>::loadSequences(fraction);
		thrust::fill(this->_outputs().begin(),this->_outputs().end(),0);

		const Cpu::int_vector& inputs=fraction.inputWords();
		v_inputwords=inputs;
		int parallelSequences=this->parallelSequences();
		int seqnum=fraction.numSequences();
		//int inputdim=fraction.inputPatternSize();
		int we_size=this->size();
		for(int i=0;i<seqnum;++i){
			int seqlength=fraction.seqInfo(i).length;
			for(int timestep=0;timestep<seqlength;++timestep){
				int wordid=inputs[timestep*parallelSequences+i];
				if(wordid==-1){
					throw std::runtime_error(std::string("OOV! in loadSequence"));
					//continue;
				}
				thrust::copy(we_weights.begin()+wordid*we_size,
					we_weights.begin()+(wordid+1)*we_size,
					this->_outputs().begin()+we_size*(timestep*parallelSequences+i));
			}
		}
    }

	template <typename TDevice>
    void WeLayer<TDevice>::computeForwardPass()
    {
	}

	template <typename TDevice>
    void WeLayer<TDevice>::computeBackwardPass()
    {
		int we_size=this->size();
		saxpy_functor fn(n_learningrate);
        // update the we_weights
		int nn=v_inputwords.size();
		int ne=this->outputErrors().size();
		int nw=we_weights.size();
		for(int i=0;i<v_inputwords.size();++i){
			try{
			int wordid=v_inputwords[i];
			if(wordid<0){
				continue;
			}
			thrust::transform(we_weights.begin()+wordid*we_size,
				we_weights.begin()+(wordid+1)*we_size,
				this->outputErrors().begin()+i*we_size,
				we_weights.begin()+wordid*we_size,
				fn);

			}catch(...){
				int wordid=v_inputwords[i];
				int a=wordid;
				throw std::runtime_error(std::string("update we weights error!"));
			}
		}
		
    }

	template <typename TDevice>
	int WeLayer<TDevice>::vocabSize()
	{
		return vocab_size;
	}

	template <typename TDevice>
	typename TDevice::real_vector& WeLayer<TDevice>::weWeights()
	{
		return we_weights;
	}

	 template class WeLayer<Cpu>;
	 template class WeLayer<Gpu>;
}