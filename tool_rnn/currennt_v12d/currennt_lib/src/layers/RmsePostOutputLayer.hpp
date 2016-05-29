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

#ifndef LAYERS_RMSEPOSTOUTPUTLAYER_HPP
#define LAYERS_RMSEPOSTOUTPUTLAYER_HPP

#include "PostOutputLayer.hpp"


namespace layers {

    /******************************************************************************************//**
     * Implements the Root Mean Squared Error (RMSE) objective function
     *
     * RMSE = sqrt((sum(x_i-z_i)^2)/N)
     * RMSE deriv = RMSE * (x_i-z_i)
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class RmsePostOutputLayer : public PostOutputLayer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;

    private:
        real_vector m_rmses; // contains the RMSE for each pattern

    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayer The layer preceding this one
         */
        RmsePostOutputLayer(
            const helpers::JsonValue &layerChild, 
            TrainableLayer<TDevice> &precedingLayer
            );

        /**
         * Destructs the Layer
         */
        virtual ~RmsePostOutputLayer();

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


#endif // LAYERS_RMSEPOSTOUTPUTLAYER_HPP
