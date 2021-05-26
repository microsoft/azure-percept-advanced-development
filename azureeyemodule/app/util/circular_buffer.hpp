/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * This file contains a generic, thread-safe implementation of a circular buffer.
 * It is taken (with heavy modifications) from https://github.com/embeddedartistry/embedded-resources/
 * under a CC0-1.0 license.
 */
#pragma once

// Standard library includes
#include <condition_variable>
#include <mutex>
#include <vector>

// Local includes
#include "timing.hpp"
#include "helper.hpp"

namespace circbuf {

/** This class is a thread-safe version of a circular buffer. */
template <class T>
class CircularBuffer
{
public:

    /** Constructs a circular buffer of the given maximum size. */
    explicit CircularBuffer(size_t size)
        : buffer(size), read_index(0), write_index(0), max_size(size), full(false)
    {
    }

    /** Constructs a circular buffer with a default starting element. */
    explicit CircularBuffer(size_t size, T item)
        : buffer(size), read_index(0), write_index(0), max_size(size), full(false)
    {
        this->put(item);
    }

    /** Returns the capacity of the buffer. */
    size_t capacity() const;

    /** Clear the buffer of all contents. */
    void clear();

    /** Is the buffer empty? */
    bool is_empty();

    /** Is the buffer full? */
    bool is_full() const;

    /** Retrieves the next item (moving it out, not copying it). Blocks until we have something to give you. */
    T get();

    /**
     * Attempts to retrieve the next item (moving it out, not copying it) into the given reference. Blocks for up to
     * the given timeout (ms) and returns true if we read something, false otherwise (in which case, the object
     * will not be filled in).
     *
     * This method takes an internal mutex until the timeout_ms expires or until we have something.
     */
    bool get_with_timeout(T &item, unsigned long int timeout_ms);

    /**
     * Attempts to retrieve the next item (moving it out, not copying it) into the given reference. Does not block.
     * Returns true if successful, false otherwise.
     *
     * This method only takes the internal mutex if we have something to give, and only for as long as it takes to retrieve
     * the item. It never waits on the mutex - if the mutex is already taken, we return false, regardless of whether
     * there is something to give or not. Therefore, this is safe to call from an interrupt context.
     */
    bool get_no_wait(T &item);

    /** Returns the current number of items in the buffer. Requires taking the mutex. */
    size_t size();

    /** Returns the best estimate of the current number of items in the buffer. Could be wrong though because it does not take the mutex. */
    size_t size_no_wait() const;

    /** Puts the given item into the buffer. This will use a copy constructor, so try to use std::move if you can. */
    void put(T item);

private:

    /** We use a mutex as our atomic-enforcing mechanism to make this thread safe. */
    std::mutex mutex;

    /** This is used to allow synchronization between the get() and put() methods without busy waiting. */
    std::condition_variable condition;

    /** We use a vector as the underlying buffer. We are simply adding some stuff on top of it. */
    std::vector<T> buffer;

    /** This is where we read from. */
    size_t read_index;

    /** This is where we write to. */
    size_t write_index;

    /** This is the largest that our circular buffer is allowed to grow to. */
    const size_t max_size;

    /** Are we full? */
    bool full;

    /**
     * Returns a best estimate as to whether we are empty. Note that it may be wrong because
     * the check for emptiness is not atomic. The public-facing API call `is_empty()` checks
     * by first grabbing the mutex before doing the check, but this function is useful internally
     * when we already have the mutex and need to check if we are empty.
     */
    bool is_empty_not_threadsafe() const;
};

template<class T>
size_t CircularBuffer<T>::capacity() const
{
    return this->max_size;
}

template<class T>
void CircularBuffer<T>::clear()
{
    this->mutex.lock();

    this->read_index = this->write_index;
    this->full = false;

    this->mutex.unlock();
}

template<class T>
bool CircularBuffer<T>::is_empty()
{
    this->mutex.lock();
    bool empty = this->is_empty_not_threadsafe();
    this->mutex.unlock();

    return empty;
}

template<class T>
bool CircularBuffer<T>::is_full() const
{
    return this->full;
}

template<class T>
T CircularBuffer<T>::get()
{
    std::unique_lock<std::mutex> lock(this->mutex);

    if (this->is_empty_not_threadsafe())
    {
        // Sleep this thread until we get signaled by put() that something is present.
        // Note that we would like to call !this->is_empty(), but since is_empty acquires the mutex,
        // we can't do that. But, when we are woken spuriously, we are guaranteed to reacquire the mutex
        // before checking the predicate, so there is no race condition on checking !is_empty_not_threadsafe().
        // Therefore, when we wake up and find that the predicate is true (i.e., we are not empty),
        // the rest of this method is guaranteed to act under that same context - i.e., that we are not empty,
        // since we have reacquired the mutex.
        this->condition.wait(lock, [&]{return !this->is_empty_not_threadsafe();});
    }

    auto val = std::move(this->buffer.at(this->read_index));
    this->full = false;
    this->read_index = (this->read_index + 1) % this->max_size;

    lock.unlock();
    return val;
}

template<class T>
bool CircularBuffer<T>::get_with_timeout(T &item, unsigned long int timeout_ms)
{
    std::unique_lock<std::mutex> lock(this->mutex);

    if (this->is_empty_not_threadsafe())
    {
        // Sleep this thread until either we get signaled by put() or we wake up due to timeout.
        this->condition.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]{return !this->is_empty_not_threadsafe();});
    }

    if (!this->is_empty_not_threadsafe())
    {
        // Read out the next item
        item = std::move(this->buffer.at(this->read_index));
        this->full = false;
        this->read_index = (this->read_index + 1) % this->max_size;

        // Unlock and return true to show that we got an item.
        lock.unlock();
        return true;
    }
    else
    {
        // We were woken up by the alarm, so we timed out. Unlock and return false.
        lock.unlock();
        return false;
    }
}

template<class T>
bool CircularBuffer<T>::get_no_wait(T &item)
{
    if (this->mutex.try_lock())
    {
        // We now have the mutex. Take an item if we have one, otherwise return false.
        if (this->is_empty_not_threadsafe())
        {
            this->mutex.unlock();
            return false;
        }
        else
        {
            // Read out the next item
            item = std::move(this->buffer.at(this->read_index));
            this->full = false;
            this->read_index = (this->read_index + 1) % this->max_size;

            // Unlock and return true to show that we got an item.
            this->mutex.unlock();
            return true;
        }
    }
    else
    {
        // We failed to get the mutex. Return false.
        return false;
    }
}

template<class T>
void CircularBuffer<T>::put(T item)
{
    this->mutex.lock();

    this->buffer[this->write_index] = item;
    if (this->full)
    {
        // Since we determine if we are full by checking if read == write,
        // we need to keep them in sync when we are full (until something gets read).
        this->read_index = (this->read_index + 1) % this->max_size;
    }

    this->write_index = (this->write_index + 1) % this->max_size;
    this->full = (this->write_index == this->read_index);

    this->mutex.unlock();

    // Signal someone if anyone is blocking on get()
    this->condition.notify_one();
}

template<class T>
size_t CircularBuffer<T>::size()
{
    this->mutex.lock();
    auto size = this->size_no_wait();
    this->mutex.unlock();

    return size;
}

template<class T>
size_t CircularBuffer<T>::size_no_wait() const
{
    size_t size = this->max_size;

    if (!this->full)
    {
        auto write = this->write_index;
        auto read = this->read_index;
        if (write >= read)
        {
            size = write - read;
        }
        else
        {
            size = this->max_size + write - read;
        }
    }

    return size;
}

template<class T>
bool CircularBuffer<T>::is_empty_not_threadsafe() const
{
    // If the place where we are reading from is the same as where
    // we are about to write to, we are considered full or empty.
    return !this->full && (this->read_index == this->write_index);
}

} // namespace circbuf
