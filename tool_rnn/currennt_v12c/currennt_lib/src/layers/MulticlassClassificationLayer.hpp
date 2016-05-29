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

#ifndef LAYERS_MULTICLASSCLASSIFICATIONLAYER_HPP
#define LAYERS_MULTICLASSCLASSIFICATIONLAYER_HPP

#include "PostOutputLayer.hpp"


namespace layers {

    /******************************************************************************************//**
     * Post output layer for multiclass classification
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class MulticlassClassificationLayer : public PostOutputLayer<TDevice>
    {
        typedef typename TDevice::int_vector int_vector;

    private:
        int_vector m_patTargetClasses;

    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayer The layer preceding this one
         */
        MulticlassClassificationLayer(
            const helpers::JsonValue &layerChild, 
            TrainableLayer<TDevice>  &precedingLayer
            );

        /**
         * Destructs the Layer
         */
        virtual ~MulticlassClassificationLayer();

        /**
         * Counts correct classifications
         *
         * @return Number of correct classifications
         */
        int countCorrectClassifications();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see Layer::loadSequences()
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

        /**
         * @see PostOutputLayer::calculateError()
         */
        virtual real_t calculateError();

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass();

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass();
    };

} // namespace layers


#endif // LAYERS_MULTICLASSCLASSIFICATIONLAYER_HPP
