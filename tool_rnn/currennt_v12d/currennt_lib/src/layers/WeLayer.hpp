
#ifndef LAYERS_WeeLayer_HPP
#define LAYERS_WeeLayer_HPP

#include "TrainableLayer.hpp"

namespace layers {

template <typename TDevice>
class WeLayer : public TrainableLayer<TDevice>
{
	typedef typename TDevice::real_vector real_vector;

	//typedef typename TDevice::real_vector real_vector;
	typename TDevice::real_vector we_weights;
	Cpu::int_vector v_inputwords;
	real_t n_learningrate;
	int vocab_size;
	
public:
    /**
        * Constructs the Layer
        *
        * @param layerChild     The layer child of the JSON configuration for this layer
        * @param weightsSection The weights section of the JSON configuration
        * @param precedingLayer The layer preceding this one
        */
    WeLayer(
        const helpers::JsonValue &layerChild, 
        const helpers::JsonValue &weightsSection,
        Layer<TDevice>           &precedingLayer
        );

    /**
        * Destructs the Layer
        */
    virtual ~WeLayer();

    /**
        * @see Layer::type()
        */
    virtual const std::string& type() const;

    /**
        * @see Layer::computeForwardPass()
        */
    virtual void computeForwardPass();

        /**
        * @see Layer::computeBackwardPass()
        */
    virtual void computeBackwardPass();
	
	virtual void loadSequences(const data_sets::DataSetFraction &fraction);

	int vocabSize();

	real_vector& weWeights();
	
};

} // namespace layers


#endif // LAYERS_WeLayer_HPP