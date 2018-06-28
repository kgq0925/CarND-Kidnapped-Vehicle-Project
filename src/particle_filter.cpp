/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 500;

	particles.reserve(num_particles);
	weights.reserve(num_particles);

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for(int i = 0; i < num_particles; ++i) {
		Particle pt;
		pt.id = i;
		pt.x = dist_x(gen);
		pt.y = dist_y(gen);
		pt.theta = dist_theta(gen);
		pt.weight = 1.0f;
		particles.push_back(pt);
		weights.push_back(pt.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
  	for (auto &particle : particles) {
      	double x = particle.x;
      	double y = particle.y;
      	double theta = particle.theta;

      	if (fabs(yaw_rate) < 0.000001) {
          	x += velocity * delta_t * cos(theta);
          	y += velocity * delta_t * sin(theta);
      	}
		else {
          	x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
          	y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      	}
      	theta += yaw_rate * delta_t;

      	normal_distribution<double> dist_x(x, std_pos[0]);
      	normal_distribution<double> dist_y(y, std_pos[1]);
      	normal_distribution<double> dist_theta(theta, std_pos[2]);

      	particle.x = dist_x(gen);
      	particle.y = dist_y(gen);
      	particle.theta = dist_theta(gen);
  	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto &obs : observations) {
    	double distance = std::numeric_limits<double>::max();
    	for (auto pred : predicted) {
      		double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);
      		if (cur_dist < distance ) {
        		distance = cur_dist;
        		obs.id = pred.id;
      		}
    	}
  	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	std::vector<LandmarkObs> pred;
	pred.reserve(map_landmarks.landmark_list.size());
	for(auto landmark : map_landmarks.landmark_list) {
		LandmarkObs obs;
		obs.x = landmark.x_f;
		obs.y = landmark.y_f;
		obs.id = landmark.id_i;
		pred.push_back(obs);
	}

  	for (int i = 0; i < num_particles; ++i) {
    	// covert current observations to global coordinates
    	std::vector<LandmarkObs> global_obs;
    	global_obs.reserve(observations.size());

		for(auto observation : observations) {
			LandmarkObs obs;
			obs.x = particles[i].x + observation.x * cos(particles[i].theta) - observation.y * sin(particles[i].theta);
			obs.y = particles[i].y + observation.x * sin(particles[i].theta) + observation.y * cos(particles[i].theta);
			obs.id = -1;
			global_obs.push_back(obs);
		}

    	dataAssociation(pred, global_obs);

    	// update weights
    	weights[i] = 1.0f;
    	for(auto obs : global_obs) {
      		weights[i] *= 1.0f / (2.0f * M_PI * std_landmark[0] * std_landmark[1]) * exp(-1.0f * (pow(pred[obs.id - 1].x - obs.x, 2) / (2.0f * pow(std_landmark[0], 2)) + pow(pred[obs.id - 1].y - obs.y, 2) / (2.0f * pow(std_landmark[1], 2))));
    	}

    	particles[i].weight = weights[i];
  	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> resampled;
    std::default_random_engine gen;
    std::discrete_distribution<> distribution(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; ++i) {
        resampled.push_back(particles[distribution(gen)]);
    }
    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
