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

#ifndef LAYERS_BINARYCLASSIFICATIONLAYER_HPP
#define LAYERS_BINARYCLASSIFICATIONLAYER_HPP

#include "PostOutputLayer.hpp"


namespace layers {

    /******************************************************************************************//**
     * Post output layer for binary classification
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class BinaryClassificationLayer : public PostOutputLayer<TDevice>
    {
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayer The layer preceding this one
         */
        BinaryClassificationLayer(
            const helpers::JsonValue &layerChild, 
            TrainableLayer<TDevice>  &precedingLayer
            );

        /**
         * Destructs the Layer
         */
        virtual ~BinaryClassificationLayer();

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


#endif // LAYERS_BINARYCLASSIFICATIONLAYER_HPP
