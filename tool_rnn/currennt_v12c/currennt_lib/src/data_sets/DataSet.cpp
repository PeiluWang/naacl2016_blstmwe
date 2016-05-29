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

#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <boost/function.hpp>

#include "DataSet.hpp"
#include "../Configuration.hpp"

#include "../netcdf/netcdf.h"

#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cassert>


namespace {
namespace internal {

    int readNcDimension(int ncid, const char *dimName)
    {
        int ret;
        int dimid;
        size_t x;

        if ((ret = nc_inq_dimid(ncid, dimName, &dimid)) || (ret = nc_inq_dimlen(ncid, dimid, &x)))
            throw std::runtime_error(std::string("Cannot get dimension '") + dimName + "': " + nc_strerror(ret));

        return (int)x;
    }

    bool hasNcDimension(int ncid, const char *dimName)
    {
        try {
            readNcDimension(ncid, dimName);
            return true;
        } 
        catch (...) {
            return false;
        }
    }

    std::string readNcStringArray(int ncid, const char *arrName, int arrIdx, int maxStringLength)
    {
        int ret;
        int varid;
        char *buffer = new char[maxStringLength+1];
        size_t start[] = {arrIdx, 0};
        size_t count[] = {1, maxStringLength};

        if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_text(ncid, varid, start, count, buffer)))
            throw std::runtime_error(std::string("Cannot read variable '") + arrName + "': " + nc_strerror(ret));

        buffer[maxStringLength] = '\0';
        return std::string(buffer);
    }

    int readNcIntArray(int ncid, const char *arrName, int arrIdx)
    {
        int ret;
        int varid;
        size_t start[] = {arrIdx};
        size_t count[] = {1};

        int x;
        if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_int(ncid, varid, start, count, &x)))
            throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

        return x;
    }

    template <typename T>
    int _readNcArrayHelper(int ncid, int varid, const size_t start[], const size_t count[], T *v);

    template <>
    int _readNcArrayHelper<float>(int ncid, int varid, const size_t start[], const size_t count[], float *v)
    {
        return nc_get_vara_float(ncid, varid, start, count, v);
    }

    template <>
    int _readNcArrayHelper<double>(int ncid, int varid, const size_t start[], const size_t count[], double *v)
    {
        return nc_get_vara_double(ncid, varid, start, count, v);
    }

    template <>
    int _readNcArrayHelper<int>(int ncid, int varid, const size_t start[], const size_t count[], int *v)
    {
        return nc_get_vara_int(ncid, varid, start, count, v);
    }

    template <typename T>
    thrust::host_vector<T> readNcArray(int ncid, const char *arrName, int begin, int n)
    {
        int ret;
        int varid;
        size_t start[] = {begin};
        size_t count[] = {n};

        thrust::host_vector<T> v(n);
        if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = _readNcArrayHelper<T>(ncid, varid, start, count, v.data())))
            throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

        return v;
    }

    Cpu::real_vector readNcPatternArray(int ncid, const char *arrName, int begin, int n, int patternSize)
    {
        int ret;
        int varid;
        size_t start[] = {begin, 0};
        size_t count[] = {n, patternSize};

        Cpu::real_vector v(n * patternSize);
        if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = _readNcArrayHelper<real_t>(ncid, varid, start, count, v.data())))
            throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

        return v;
    }

    Cpu::real_vector targetClassesToOutputs(const Cpu::int_vector &targetClasses, int numLabels)
    {
        if (numLabels == 2) {
            Cpu::real_vector v(targetClasses.size());
            for (size_t i = 0; i < v.size(); ++i)
                v[i] = (real_t)targetClasses[i];

            return v;
        }
        else {
            Cpu::real_vector v(targetClasses.size() * numLabels, 0);

            for (size_t i = 0; i < targetClasses.size(); ++i)
                v[i * numLabels + targetClasses[i]] = 1;

            return v;
        }
    }

    bool comp_seqs(const data_sets::DataSet::sequence_t &a, const data_sets::DataSet::sequence_t &b)
    {
        return (a.length < b.length);
    }

    struct rand_gen {
        unsigned operator()(unsigned i)
        {
            static boost::mt19937 *gen = NULL;
            if (!gen) {
                gen = new boost::mt19937;
                gen->seed(Configuration::instance().randomSeedShuffle());
            }

            boost::uniform_int<> dist(0, i-1);
            return dist(*gen);
        }
    };

} // namespace internal
} // anonymous namespace


namespace data_sets {

    struct thread_data_t
    {
        boost::thread             thread;
        boost::mutex              mutex;
        boost::condition_variable cv;
        bool                      terminate;
        
        boost::function<boost::shared_ptr<DataSetFraction> ()> taskFn;
        boost::shared_ptr<DataSetFraction> frac;
        bool finished;
    };

    void DataSet::_nextFracThreadFn()
    {
        for (;;) {
            // wait for a new task
            boost::unique_lock<boost::mutex> lock(m_threadData->mutex);
            while (m_threadData->taskFn.empty() && !m_threadData->terminate)
                m_threadData->cv.wait(lock);

            // terminate the thread?
            if (m_threadData->terminate)
                break;

            // execute the task
            m_threadData->frac.reset();
            m_threadData->frac = m_threadData->taskFn();
            m_threadData->finished = true;
            m_threadData->taskFn.clear();

            // tell the others that we are ready
            m_threadData->cv.notify_one();
        }
    }

    void DataSet::_shuffleSequences()
    {
        internal::rand_gen rg;
        std::random_shuffle(m_sequences.begin(), m_sequences.end(), rg);
    }

    void DataSet::_shuffleFractions()
    {
        std::vector<std::vector<sequence_t> > fractions;
        for (size_t i = 0; i < m_sequences.size(); ++i) {
            if (i % m_parallelSequences == 0)
                fractions.resize(fractions.size() + 1);
            fractions.back().push_back(m_sequences[i]);
        }

        internal::rand_gen rg;
        std::random_shuffle(fractions.begin(), fractions.end(), rg);

        m_sequences.clear();
        for (size_t i = 0; i < fractions.size(); ++i) {
            for (size_t j = 0; j < fractions[i].size(); ++j)
                m_sequences.push_back(fractions[i][j]);
        }
    }

    void DataSet::_addNoise(Cpu::real_vector *v)
    {
        if (!m_noiseDeviation)
            return;

        static boost::mt19937 *gen = NULL;
        if (!gen) {
            gen = new boost::mt19937;
            gen->seed(Configuration::instance().randomSeed());
        }

        boost::normal_distribution<real_t> dist((real_t)0, m_noiseDeviation);

        for (size_t i = 0; i < v->size(); ++i)
            (*v)[i] += dist(*gen);
    }

    Cpu::real_vector DataSet::_loadInputFeatsFromCache(const sequence_t &seq)
    {
        Cpu::real_vector v(seq.length * m_inputFeatDim);

        m_cacheFile.seekg(seq.inputFeatsBegin);
        m_cacheFile.read((char*)v.data(), sizeof(real_t) * v.size());
        assert (m_cacheFile.tellg() - seq.inputFeatsBegin == v.size() * sizeof(real_t));

        return v;
    }

	Cpu::int_vector DataSet::_loadInputWordsFromCache(const sequence_t &seq)
    {
        Cpu::int_vector v(seq.length);

        m_cacheFile.seekg(seq.inputWordsBegin);
        m_cacheFile.read((char*)v.data(), sizeof(int) * v.size());
        assert (m_cacheFile.tellg() - seq.inputWordsBegin == v.size() * sizeof(int));

        return v;
    }

    Cpu::real_vector DataSet::_loadOutputsFromCache(const sequence_t &seq)
    {
        Cpu::real_vector v(seq.length * m_outputDim);

        m_cacheFile.seekg(seq.outputsBegin);
        m_cacheFile.read((char*)v.data(), sizeof(real_t) * v.size());
        assert (m_cacheFile.tellg() - seq.outputsBegin == v.size() * sizeof(real_t));

        return v;
    }

    Cpu::int_vector DataSet::_loadOutputLabelsFromCache(const sequence_t &seq)
    {
        Cpu::int_vector v(seq.length);

        m_cacheFile.seekg(seq.outputsBegin);
        m_cacheFile.read((char*)v.data(), sizeof(int) * v.size());
        assert (m_cacheFile.tellg() - seq.outputsBegin == v.size() * sizeof(int));

        return v;
    }

    boost::shared_ptr<DataSetFraction> DataSet::_makeFractionTask(int firstSeqIdx)
    {
        //printf("(%d) Making task firstSeqIdx=%d...\n", (int)m_sequences.size(), firstSeqIdx);
        boost::shared_ptr<DataSetFraction> frac(new DataSetFraction);
        frac->m_inputFeatDim  = m_inputFeatDim;
		frac->m_outputDim = m_outputDim;
        frac->m_maxSeqLength      = std::numeric_limits<int>::min();
        frac->m_minSeqLength      = std::numeric_limits<int>::max();

        // fill fraction sequence info
        for (int seqIdx = firstSeqIdx; seqIdx < firstSeqIdx + m_parallelSequences; ++seqIdx) {
            if (seqIdx < (int)m_sequences.size()) {
                frac->m_maxSeqLength = std::max(frac->m_maxSeqLength, m_sequences[seqIdx].length);
                frac->m_minSeqLength = std::min(frac->m_minSeqLength, m_sequences[seqIdx].length);

                DataSetFraction::seq_info_t seqInfo;
                seqInfo.originalSeqIdx = m_sequences[seqIdx].originalSeqIdx;
                seqInfo.length         = m_sequences[seqIdx].length;
                seqInfo.seqTag         = m_sequences[seqIdx].seqTag;

                frac->m_seqInfo.push_back(seqInfo);
            }
        }

        // allocate memory for the fraction
        frac->m_inputFeats  .resize(frac->m_maxSeqLength * m_parallelSequences * m_inputFeatDim, 0);
		frac->m_inputWords  .resize(frac->m_maxSeqLength * m_parallelSequences, -1);
        frac->m_patTypes.resize(frac->m_maxSeqLength * m_parallelSequences, PATTYPE_NONE);

        if (m_isClassificationData)
            frac->m_outputLabels.resize(frac->m_maxSeqLength * m_parallelSequences, -1);
        else
            frac->m_outputs.resize(frac->m_maxSeqLength * m_parallelSequences * m_outputDim);

		int vocab_size=Configuration::instance().vocabSize();
        // load sequences from the cache file and create the fraction vectors
        for (int i = 0; i < m_parallelSequences; ++i) {
            if (firstSeqIdx + i >= (int)m_sequences.size())
                continue;

            const sequence_t &seq = m_sequences[firstSeqIdx + i];

            // input feats
            Cpu::real_vector inputFeats = _loadInputFeatsFromCache(seq);
            _addNoise(&inputFeats);
            for (int timestep = 0; timestep < seq.length; ++timestep) {
				int s_start = m_inputFeatDim * timestep;
				int t_start = m_inputFeatDim * (timestep * m_parallelSequences + i);
				thrust::copy_n(inputFeats.begin()+s_start, m_inputFeatDim, frac->m_inputFeats.begin()+t_start);
            }
			// input words
			Cpu::int_vector inputWords = _loadInputWordsFromCache(seq);
            for (int timestep = 0; timestep < seq.length; ++timestep) {
				assert(inputWords[timestep] < vocab_size);
				frac->m_inputWords[timestep * m_parallelSequences + i] = inputWords[timestep];
            }
            // output labels
            if (m_isClassificationData) {
                Cpu::int_vector outputLabels = _loadOutputLabelsFromCache(seq);
                for (int timestep = 0; timestep < seq.length; ++timestep)
                    frac->m_outputLabels[timestep * m_parallelSequences + i] = outputLabels[timestep];
            }else {
				// output values
				Cpu::real_vector outputs = _loadOutputsFromCache(seq);
                for (int timestep = 0; timestep < seq.length; ++timestep) {
                    int s_start = m_outputDim * timestep;
                    int t_start = m_outputDim * (timestep * m_parallelSequences + i);
                    thrust::copy_n(outputs.begin() + s_start, m_outputDim, frac->m_outputs.begin() + t_start);
                }
            }
            // pattern types
            for (int timestep = 0; timestep < seq.length; ++timestep) {
                Cpu::pattype_vector::value_type patType;
                if (timestep == 0)
                    patType = PATTYPE_FIRST;
                else if (timestep == seq.length - 1)
                    patType = PATTYPE_LAST;
                else
                    patType = PATTYPE_NORMAL;

                frac->m_patTypes[timestep * m_parallelSequences + i] = patType;
            }
        }
        return frac;
    }

    boost::shared_ptr<DataSetFraction> DataSet::_makeFirstFractionTask()
    {
        //printf("(%d) Making first task...\n", (int)m_sequences.size());
        
        if (m_sequenceShuffling)
            _shuffleSequences();
        if (m_fractionShuffling)
            _shuffleFractions();

        return _makeFractionTask(0);
    }

    DataSet::DataSet()
        : m_fractionShuffling(false)
        , m_sequenceShuffling(false)
        , m_noiseDeviation   (0)
        , m_parallelSequences(0)
        , m_totalSequences   (0)
        , m_totalTimesteps   (0)
        , m_minSeqLength     (0)
        , m_maxSeqLength     (0)
        , m_inputFeatDim     (0)
        , m_outputDim(0)
        , m_curFirstSeqIdx   (-1)
    {
    }

    DataSet::DataSet(const std::string &ncfile, int parSeq, real_t fraction, bool fracShuf, bool seqShuf, real_t noiseDev)
        : m_fractionShuffling(fracShuf)
        , m_sequenceShuffling(seqShuf)
        , m_noiseDeviation   (noiseDev)
        , m_parallelSequences(parSeq)
        , m_totalTimesteps   (0)
        , m_minSeqLength     (std::numeric_limits<int>::max())
        , m_maxSeqLength     (std::numeric_limits<int>::min())
        , m_curFirstSeqIdx   (-1)
    {
        int ret;
        int ncid;

        if (fraction <= 0 || fraction > 1)
            throw std::runtime_error("Invalid fraction");

        // open the cache file
        std::string tmpFileName = (boost::filesystem::temp_directory_path() / boost::filesystem::unique_path()).string();
		if(!Configuration::instance().cacheDir().empty()){
			tmpFileName=Configuration::instance().cacheDir()+"\\"+boost::filesystem::unique_path().string();
		}
        m_cacheFile.open(tmpFileName.c_str(), std::fstream::in | std::fstream::out | std::fstream::binary | std::fstream::trunc);
        if (!m_cacheFile.good())
            throw std::runtime_error(std::string("Cannot open temporary file '") + tmpFileName + "'");

        // open the *.nc file
        if ((ret = nc_open(ncfile.c_str(), NC_NOWRITE, &ncid)))
            throw std::runtime_error(std::string("Could not open '") + ncfile + "': " + nc_strerror(ret));

        // extract the patterns from the *.nc file
        try {
			m_isClassificationData = !internal::hasNcDimension(ncid,"outputValues");
            int maxSeqTagLength    = internal::readNcDimension(ncid, "seqname_maxlength");
            m_inputFeatDim     = internal::readNcDimension(ncid, "feat_dim");
			m_outputDim = internal::readNcDimension(ncid, "output_dim");
            m_totalSequences = internal::readNcDimension(ncid, "seq_num");

			if(m_isClassificationData){
				std::cout<<"\nClassification task\n"<<std::endl;
			}else{
				std::cout<<"\nRegression task\n"<<std::endl;
			}
            
			assert(m_totalSequences>1);
			
            int inputFeatsBegin = 0;
			int inputWordsBegin = 0;
            int targetsBegin = 0;
            for (int i = 0; i < m_totalSequences; ++i) {
                int seqLength = internal::readNcIntArray(ncid, "seqLengths", i);

                m_totalTimesteps += seqLength;
                m_minSeqLength = std::min(m_minSeqLength, seqLength);
                m_maxSeqLength = std::max(m_maxSeqLength, seqLength);

                sequence_t seq;
                seq.originalSeqIdx = i;
                seq.length         = seqLength;
                seq.seqTag         = internal::readNcStringArray(ncid, "seqTags", i, maxSeqTagLength);

                // read input feats and store them in the cache file
                seq.inputFeatsBegin = m_cacheFile.tellp();
                Cpu::real_vector inputFeats = internal::readNcPatternArray(ncid, "inputFeats", inputFeatsBegin, seqLength, m_inputFeatDim);
                m_cacheFile.write((const char*)inputFeats.data(), sizeof(real_t) * inputFeats.size());
                assert (m_cacheFile.tellp() - seq.inputFeatsBegin == seqLength * m_inputFeatDim * sizeof(real_t));

				// read input words and store them in the cache file
                seq.inputWordsBegin = m_cacheFile.tellp();
                Cpu::int_vector inputWords = internal::readNcArray<int>(ncid, "inputWords", inputWordsBegin, seqLength);
                m_cacheFile.write((const char*)inputWords.data(), sizeof(int) * inputWords.size());
                assert (m_cacheFile.tellp() - seq.inputWordsBegin == seqLength * sizeof(int));

                // read targets and store them in the cache file
                seq.outputsBegin = m_cacheFile.tellp();
                if (m_isClassificationData) {
                    Cpu::int_vector targets = internal::readNcArray<int>(ncid, "outputLabels", targetsBegin, seqLength);
                    m_cacheFile.write((const char*)targets.data(), sizeof(int) * targets.size());
                    assert (m_cacheFile.tellp() - seq.outputsBegin == seqLength * sizeof(int));
                }
                else {
                    Cpu::real_vector targets = internal::readNcPatternArray(ncid, "outputValues", targetsBegin, seqLength, m_outputDim);
                    m_cacheFile.write((const char*)targets.data(), sizeof(real_t) * targets.size());
                    assert (m_cacheFile.tellp() - seq.outputsBegin == seqLength * m_outputDim * sizeof(real_t));
                }

                m_sequences.push_back(seq);

				inputFeatsBegin  += seqLength;
                inputWordsBegin  += seqLength;
                targetsBegin += seqLength;
            }

            // sort sequences by length
            if (Configuration::instance().trainingMode()){
				if(Configuration::instance().sortTrainData()){
					printf("\n> sort train sequences\n");
					std::sort(m_sequences.begin(), m_sequences.end(), internal::comp_seqs);
				}
			}

            // create next fraction data and start the thread
            m_threadData.reset(new thread_data_t);
            m_threadData->finished  = false;
            m_threadData->terminate = false;
            m_threadData->thread    = boost::thread(&DataSet::_nextFracThreadFn, this);
        }
        catch (const std::exception&) {
            nc_close(ncid);
            throw;
        }
    }

    DataSet::~DataSet()
    {
        // terminate the next fraction thread
        if (m_threadData) {
            {{
                boost::lock_guard<boost::mutex> lock(m_threadData->mutex);
                m_threadData->terminate = true;
                m_threadData->cv.notify_one();
            }}

            m_threadData->thread.join();
        }
    }

    bool DataSet::isClassificationData() const
    {
        return m_isClassificationData;
    }

    bool DataSet::empty() const
    {
        return (m_totalTimesteps == 0);
    }

    boost::shared_ptr<DataSetFraction> DataSet::getNextFraction()
    {
        // initial work
        if (m_curFirstSeqIdx == -1) {
            boost::unique_lock<boost::mutex> lock(m_threadData->mutex);
            m_threadData->taskFn = boost::bind(&DataSet::_makeFirstFractionTask, this);
            m_threadData->finished = false;
            m_threadData->cv.notify_one();
            m_curFirstSeqIdx = 0;
        }

        // wait for the thread to finish
        boost::unique_lock<boost::mutex> lock(m_threadData->mutex);
        while (!m_threadData->finished)
            m_threadData->cv.wait(lock);

        // get the fraction
        boost::shared_ptr<DataSetFraction> frac;
        if (m_curFirstSeqIdx < (int)m_sequences.size()) {
            frac = m_threadData->frac;
            m_curFirstSeqIdx += m_parallelSequences;

            // start new task
            if (m_curFirstSeqIdx < (int)m_sequences.size())
                m_threadData->taskFn = boost::bind(&DataSet::_makeFractionTask, this, m_curFirstSeqIdx);
            else
                m_threadData->taskFn = boost::bind(&DataSet::_makeFirstFractionTask, this);

            m_threadData->finished = false;
            m_threadData->cv.notify_one();
        }
        else  {
            m_curFirstSeqIdx = 0;
        }

        return frac;
    }

    int DataSet::totalSequences() const
    {
        return m_totalSequences;
    }

    int DataSet::totalTimesteps() const
    {
        return m_totalTimesteps;
    }

    int DataSet::minSeqLength() const
    {
        return m_minSeqLength;
    }

    int DataSet::maxSeqLength() const
    {
        return m_maxSeqLength;
    }

    int DataSet::inputFeatDim() const
    {
        return m_inputFeatDim;
    }

    int DataSet::outputDim() const
    {
        return m_outputDim;
    }

} // namespace data_sets
