//
// Copyright (c) 2012 Daniel Lundin
//

#include "medida/stats/exp_decay_sample.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <mutex>
#include <random>

#include "medida/stats/snapshot.h"

namespace medida
{
namespace stats
{

static const Clock::duration kRESCALE_THRESHOLD = std::chrono::minutes{6};
static const double BLEED_TIME_SECONDS = 30;
static const double SNAPSHOT_WINDOW_SECONDS = 60;

class ExpDecaySample::Impl
{
  public:
    Impl(std::uint32_t reservoirSize, double alpha);
    ~Impl();
    void Clear();
    std::uint64_t size() const;
    void Update(std::int64_t value);
    void Update(std::int64_t value, Clock::time_point timestamp);
    Snapshot MakeSnapshot() const;

  private:
    const double alpha_;
    const std::uint64_t reservoirSize_;
    Clock::time_point startTime_;
    Clock::time_point nextScaleTime_;

    std::atomic<std::uint64_t> count_;
    struct WeightedValue
    {
        std::int64_t value;
        double weight;
    };
    std::map<double, WeightedValue> values_;
    std::mutex mutex_;
    mutable std::mt19937 rng_;
    std::uniform_real_distribution<> dist_;
    void Rescale(const Clock::time_point& when);
};

ExpDecaySample::ExpDecaySample(std::uint32_t reservoirSize, double alpha)
    : impl_{new ExpDecaySample::Impl{reservoirSize, alpha}}
{
}

ExpDecaySample::~ExpDecaySample()
{
}

void
ExpDecaySample::Clear()
{
    impl_->Clear();
}

std::uint64_t
ExpDecaySample::size() const
{
    return impl_->size();
}

void
ExpDecaySample::Update(std::int64_t value)
{
    impl_->Update(value);
}

void
ExpDecaySample::Update(std::int64_t value, Clock::time_point timestamp)
{
    impl_->Update(value, timestamp);
}

Snapshot
ExpDecaySample::MakeSnapshot() const
{
    return impl_->MakeSnapshot();
}

// === Implementation ===

ExpDecaySample::Impl::Impl(std::uint32_t reservoirSize, double alpha)
    : alpha_{alpha}
    , reservoirSize_{reservoirSize}
    , count_{}
    , rng_{std::random_device()()}
    , dist_(0.0, 1.0)
{
    Clear();
}

ExpDecaySample::Impl::~Impl()
{
}

void
ExpDecaySample::Impl::Clear()
{
    std::lock_guard<std::mutex> lock{mutex_};
    values_.clear();
    count_ = 0;
    startTime_ = Clock::now();
    nextScaleTime_ = startTime_ + kRESCALE_THRESHOLD;
}

std::uint64_t
ExpDecaySample::Impl::size() const
{
    return std::min(reservoirSize_, count_.load());
}

void
ExpDecaySample::Impl::Update(std::int64_t value)
{
    Update(value, Clock::now());
}

void
ExpDecaySample::Impl::Update(std::int64_t value, Clock::time_point timestamp)
{
    if (timestamp >= nextScaleTime_)
    {
        Rescale(timestamp);
    }
    std::lock_guard<std::mutex> lock{mutex_};
    auto dur = std::chrono::duration_cast<std::chrono::seconds>(timestamp -
                                                                startTime_);
    auto itemWeight = std::exp(alpha_ * dur.count());

    // priority is weight adjusted up to BLEED_TIME_SECONDS seconds into the
    // future
    auto priority = std::exp(
        alpha_ * (double(dur.count()) + BLEED_TIME_SECONDS * dist_(rng_)));
    auto count = ++count_;

    WeightedValue wv = {value, itemWeight};
    if (count <= reservoirSize_)
    {
        values_[priority] = wv;
    }
    else
    {
        auto first = std::begin(values_)->first;
        if (first < priority && values_.insert({priority, wv}).second)
        {
            values_.erase(first);
        }
    }
}

void
ExpDecaySample::Impl::Rescale(const Clock::time_point& when)
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (when < nextScaleTime_)
        return;

    nextScaleTime_ = when + kRESCALE_THRESHOLD;
    auto oldStartTime = startTime_;
    startTime_ = when;

    auto dur =
        std::chrono::duration_cast<std::chrono::seconds>(when - oldStartTime);
    auto scalingFactor = std::exp(-alpha_ * dur.count());

    if (scalingFactor == 0.0)
    {
        values_.clear();
    }
    else
    {
        std::map<double, WeightedValue> newValues;

        for (auto const& kv : values_)
        {
            auto newWeight = kv.second.weight * scalingFactor;
            if (newWeight != 0.0)
            {
                newValues[kv.first * scalingFactor] = {kv.second.value,
                                                       newWeight};
            }
        }
        std::swap(newValues, values_);
    }
    count_ = values_.size();
}

Snapshot
ExpDecaySample::Impl::MakeSnapshot() const
{
    std::vector<double> vals;

    if (values_.empty())
        return {vals};

    std::vector<WeightedValue> wvals;
    wvals.reserve(values_.size());

    double totWeight = 0;
    for (auto const& kv : values_)
    {
        wvals.emplace_back(kv.second);
        totWeight += kv.second.weight;
    }

    std::sort(wvals.begin(), wvals.end(),
              [](WeightedValue const& l, WeightedValue const& r) {
                  return l.value < r.value;
              });

    // derive percentiles
    // Snapshot could be changed to be weighted instead as to avoid this
    const int nbSamples = 101;
    auto const& first = *(wvals.begin());

    if (totWeight != 0)
    {
        vals.resize(nbSamples);
        int ins = 0;
        double percentile = first.weight;

        // fills using linear interpolation between values
        auto fillSome = [&vals, &ins](int count, double prevValue,
                                      double value) {
            for (int i = 1; i <= count; i++)
            {
                auto v = prevValue + (value - prevValue) * i / count;
                vals[ins++] = v;
            }
        };

        int lastVindex = percentile * nbSamples / totWeight;
        double prevValue = first.value;
        if (lastVindex > 0)
        {
            fillSome(lastVindex, prevValue, prevValue);
        }

        // process rest of the values
        for (auto it = ++wvals.begin(); it != wvals.end(); it++)
        {
            percentile += it->weight;
            int index = percentile * nbSamples / totWeight;
            if (index > lastVindex)
            {
                fillSome(index - lastVindex, prevValue, it->value);
                lastVindex = index;
                prevValue = it->value;
            }
        }
        // rounding error may have caused us to miss the last one
        if (ins != nbSamples)
        {
            fillSome(nbSamples - ins, prevValue, wvals.back().value);
        }
    }
    else
    {
        vals.emplace_back(first.value);
    }
    return {vals};
}

} // namespace stats
} // namespace medida
