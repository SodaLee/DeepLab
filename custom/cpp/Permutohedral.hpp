#ifndef PERMUTOHEDRAL_H
#define PERMUTOHEDRAL_H

#include <utility>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <vector>
#include <cmath>
#include <cstring>

#include "tensorflow/core/framework/tensor.h"

using std::unordered_map;
using std::function;
using std::pair;
using std::vector;
using tensorflow::Tensor;

struct Key
{
    int d, tag;
    int *val;
    int *shared;
    Key(int d, int *val, int tag)
    {
        this->d = d;
        this->val = new int[d];
        memcpy(this->val, val, d * sizeof(int));
        this->tag = tag;
        shared = new int;
        *shared = 1;
    }
    ~Key()
    {
        if(shared)
        {
            *shared -= 1;
            if(*shared == 0)
            {
                delete[] val;
                delete shared;
            }
        }
    }
    Key(Key&& k)
    {
        d = k.d;
        val = k.val;
        tag = k.tag;
        k.val = nullptr;
        shared = k.shared;
    }
    Key(const Key& k)
    {
        d = k.d;
        val = k.val;
        tag = k.tag;
        shared = k.shared;
        *shared += 1;
    }
    bool operator == (const Key& k) const
    {
        if(d != k.d) return false;
        if(tag != k.tag) return false;
        if(val == k.val) return true;
        for(int i = 0; i < d; i++)
            if(val[i] != k.val[i])
                return false;
        return true;
    }
    size_t hash() const
    {
        size_t r = 0;
        for(int i = 0; i < d; i++)
        {
            r += val[i];
            r *= 1664525;
        }
        r += tag;
        r *= 1664525;
        return r;
    }
};
struct hash_key
{
    size_t operator () (const Key& k) const
    {
        return k.hash();
    }
    size_t operator () (Key& k) const
    {
        return k.hash();
    }
};

template<typename T>
class Permutohedral
{
private:
    typedef pair<int, int> Neighbour;
    int *offset_, *rank_;
    T *barycentric_;
    vector<Neighbour> *neighbours;
    int kernel_batch, kernel_d;
    std::unordered_map<Key, int, hash_key> hash_table; 
public:
    Permutohedral();
    ~Permutohedral();
    void clear();
    void init(const T *const kernel, int d, int batch, int np);
    void compute(Tensor &output_tensor, const Tensor& unary_tensor, bool add, T weight, bool reverse);
};

template<typename T>
Permutohedral<T>::Permutohedral()
{
    offset_ = rank_ = nullptr;
    barycentric_ = nullptr;
    neighbours = nullptr;
}

template<typename T>
Permutohedral<T>::~Permutohedral()
{
    delete[] offset_;
    delete[] rank_;
    delete[] barycentric_;
    delete[] neighbours;
}

template<typename T>
void Permutohedral<T>::clear()
{
    delete[] offset_;
    delete[] rank_;
    delete[] barycentric_;
    delete[] neighbours;
    offset_ = rank_ = nullptr;
    barycentric_ = nullptr;
    neighbours = nullptr;
    hash_table.clear();
}

template<typename T>
void Permutohedral<T>::init(const T *const kernel, int d, int batch, int np)
{
    offset_ = new int[batch * (d+1) * np];
    rank_ = new int[batch * (d+1) * np];
    barycentric_ = new T[batch * (d+1) * np];
    neighbours = new vector<Neighbour>[batch];
    for(int i = 0; i < batch; i++)
        new(neighbours + i) vector<Neighbour>();


    T *scale_factor = new T[d];
    int *canonical = new int[(d+1) * (d+1)];
    int *n1 = new int[d+1];
    int *n2 = new int[d+1];
    kernel_batch = batch;
    kernel_d = d;
    T *Ex = new T[d+1];//position vector
    int *rem0 = new int[d+1];//reminder-0 point
    int *_key = new int[d+1];

    // canonical simplex matrix(transposed)
    for(int i = 0; i < d+1; i++)
    {
        for(int j = 0; j < d+1-i; j++)
            canonical[i * (d+1) + j] = i;
        for(int j = d+1-i; j < d+1; j++)
            canonical[i * (d+1) + j] = i - (d+1);
    }

	T inv_std_dev = std::sqrt(2.0 / 3.0) * (d+1);
	// Compute the diagonal part of E (p.5 in [Adams etal 2010])
	for(int i = 0; i < d; i++)
		scale_factor[i] = inv_std_dev / std::sqrt((T)(i+1) * (i+2));

    for(int b = 0; b < batch; b++)
    {
        vector<Key> keys;
        keys.clear();
        int cnt = 0;
        for(int p = 0; p < np; p++)
        {
            const T * point = &kernel[(b * np + p) * d];

            // E\vec x
            T acc = 0;
            for(int i = d; i > 0; i--)
            {
                T scaled = point[i-1] * scale_factor[i-1];
                Ex[i] = acc - i * scaled;
                acc += scaled;
            }
            Ex[0] = acc;

            // get nearest remainder-0 point
            int coord_sum = 0;
            // determine if the point is on the plane
            for(int i = 0; i < d+1; i++)
            {
                int rd = std::round(Ex[i] / (d+1));
                rem0[i] = rd * (d+1);
                coord_sum += rd;
            }

            int *_rank = &rank_[(b * np + p) * (d+1)];
            memset(_rank, 0, (d+1) * sizeof(int));
            //get simplex
            for(int i = 0; i < d+1; i++)
            {
                T diff = Ex[i] - rem0[i];
                for(int j = i+1; j < d+1; j++)
                {
                    if(diff < Ex[j] - rem0[j])
                        _rank[i]++;
                    else
                        _rank[j]++;
                }
            }

            //move towards the plane
            for(int i = 0; i < d+1; i++)
            {
                _rank[i] += coord_sum;
                if(_rank[i] < 0)
                {
                    _rank[i] += d+1;
                    rem0[i] += d+1;
                }
                else if(_rank[i] > d)
                {
                    _rank[i] -= d+1;
                    rem0[i] -= d+1;
                }
            }

            T *_barycentric = &barycentric_[(b * np + p) * (d+1)];
            memset(_barycentric, 0, (d+1) * sizeof(T));
		    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
            for(int i = 0; i < d+1; i++)
            {
                T diff = (Ex[i] - rem0[i]) / (d+1);
                _barycentric[d - _rank[i]] += diff;
                if(d - _rank[i] + 1 < d+1)
                    _barycentric[d - _rank[i] + 1] -= diff;
                else
                    _barycentric[0] -=  diff;
            }
            _barycentric[0] += 1;

            int *_offset = &offset_[(b * np + p) * (d+1)];
            for(int i = 0; i < d+1; i++)
            {
                for(int j = 0; j < d; j++)
                    _key[j] = rem0[j] + canonical[i * (d+1) + _rank[j]];
                Key key(d, _key, b);
                auto iter = hash_table.find(key);
                if(iter != hash_table.end())
                    _offset[i] = iter->second;
                else
                {
                    hash_table[key] = cnt;
                    _offset[i] = cnt;
                    keys.push_back(key);
                    cnt++;
                }
            }
        }
        int M = cnt;
        neighbours[b].reserve((d+1) * M);
        for(int j = 0; j < d+1; j++)
        {
            for(int i = 0; i < M; i++)
            {
                Key& key = keys[i];
                for(int k = 0; k < d; k++)
                {
                    n1[k] = key.val[k] - 1;
                    n2[k] = key.val[k] + 1;
                }
                n1[j] = key.val[j] + d;
                n2[j] = key.val[j] - d;
                Key k1(d, n1, b);
                Key k2(d, n2, b);
                neighbours[b].emplace_back(hash_table[k1], hash_table[k2]);
            }
        }
    }
    delete[] scale_factor;
    delete[] canonical;
    delete[] n1;
    delete[] n2;
    delete[] Ex;
    delete[] rem0;
    delete[] _key;
}

template<typename T>
void Permutohedral<T>::compute(Tensor &output_tensor, const Tensor& unary_tensor, bool add, T weight, bool reverse)
{
    int batch_size = unary_tensor.dim_size(0),
        height = unary_tensor.dim_size(1),
        width = unary_tensor.dim_size(2),
        d = unary_tensor.dim_size(3);
    int np = height * width;

    for(int b = 0; b < batch_size; b++)
    {
        int kb = b;
        if(kernel_batch == 1)
            kb = 0;
        int M = neighbours[kb].size() / (kernel_d+1);
        T *values = new T[(M+2) * d];
        T *newval = new T[(M+2) * d];
        memset(values, 0, (M+2) * d * sizeof(T));
        memset(newval, 0, (M+2) * d * sizeof(T));
        auto output = output_tensor.Slice(b, b+1).flat<T>();
        auto unary = unary_tensor.Slice(b, b+1).flat<T>();

        //splatting
        for(int p = 0; p < np; p++)
        {
            for(int j = 0; j < kernel_d+1; j++)
            {
                int o = offset_[(kb * np + p) * (kernel_d+1) + j] + 1;
                T w = barycentric_[(kb * np + p) * (kernel_d+1) + j];
                for(int k = 0; k < d; k++)
                    values[o * d + k] += w * unary(p * d + k);
            }
        }

        //bluring
        if(reverse)
        {
            for(int j = 0; j < kernel_d+1; j++)
            {
                for(int i = 0; i < M; i++)
                {
                    T *oldv = &values[(i+1) * d];
                    T *newv = &newval[(i+1) * d];
                    Neighbour n = neighbours[kb][j * M + i];
                    int n1 = n.first + 1;
                    int n2 = n.second + 1;
                    T *n1v = &values[n1 * d];
                    T *n2v = &values[n2 * d];
                    for(int k = 0; k < d; k++)
                        newv[k] = oldv[k] + 0.5 * (n1v[k] + n2v[k]);
                }
                std::swap(values, newval);
            }
        }
        else
        {
            for(int j = kernel_d; j >=0; j--)
            {
                for(int i = 0; i < M; i++)
                {
                    T *oldv = &values[(i+1) * d];
                    T *newv = &newval[(i+1) * d];
                    Neighbour n = neighbours[kb][j * M + i];
                    int n1 = n.first + 1;
                    int n2 = n.second + 1;
                    T *n1v = &values[n1 * d];
                    T *n2v = &values[n2 * d];
                    for(int k = 0; k < d; k++)
                        newv[k] = oldv[k] + 0.5 * (n1v[k] + n2v[k]);
                }
                std::swap(values, newval);
            }
        }

        T alpha = 1.0 / (1 + pow(2, -kernel_d));
        //slicing
        for(int p = 0; p < np; p++)
        {
            if(!add)
            {
                for(int k = 0; k < d; k++)
                    output(p * d + k) = 0;
            }
            for(int j = 0; j < kernel_d+1; j++)
            {
                int o = offset_[(kb * np + p) * (kernel_d+1) + j] + 1;
                T w = barycentric_[(kb * np + p) * (kernel_d+1) + j];
                for(int k = 0; k < d; k++)
                    output(p * d + k) += weight * w * values[o * d + k] * alpha;
            }
        }
        delete[] values;
        delete[] newval;
    }

}

#endif
