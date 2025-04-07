#include "params.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm> // for std::remove, std::find
#include <vector>
#include <stdexcept> // for exceptions

// trim leading and trailing whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (std::string::npos == first) {
        return str; // No content
    }
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, (last - first + 1));
}

std::vector<double> parse_double_vector(const std::string& value_str) {
    std::vector<double> vec;
    std::stringstream ss(value_str);
    std::string segment;
    double val;

    while(std::getline(ss, segment, ',')) {
        // Trim whitespace from the segment
        segment = trim(segment);
        if (!segment.empty()) {
            try {
                val = std::stod(segment);
                vec.push_back(val);
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid number format in vector: '" + segment + "'");
            } catch (const std::out_of_range& e) {
                 throw std::runtime_error("Number out of range in vector: '" + segment + "'");
            }
        }
    }
    return vec;
}

// Load parameters from a key-value file
bool load_params_from_file(const std::string& filename, Params& params) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open parameter file: " << filename << std::endl;
        return false;
    }

    std::string line;
    int line_num = 0;
    std::cout << "Reading parameters from: " << filename << std::endl;

    while (std::getline(infile, line)) {
        line_num++;
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Find the '=' delimiter
        size_t delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos) {
            std::cerr << "Warning: Skipping invalid line (no '=') in " << filename
                      << " at line " << line_num << ": " << line << std::endl;
            continue;
        }

        // Split into key and value
        std::string key = trim(line.substr(0, delimiter_pos));
        std::string value_str = trim(line.substr(delimiter_pos + 1));

        if (key.empty() || value_str.empty()) {
             std::cerr << "Warning: Skipping invalid line (empty key or value) in " << filename
                       << " at line " << line_num << ": " << line << std::endl;
            continue;
        }

        // Assign value to the struct member
        try {
            if (key == "Ek") params.Ek = std::stod(value_str);
            else if (key == "Pr") params.Pr = std::stod(value_str);
            else if (key == "Pm") params.Pm = std::stod(value_str);
            else if (key == "elsm") params.elsm = parse_double_vector(value_str); 
            else if (key == "delta") params.delta = std::stod(value_str);
            else if (key == "m") params.m = std::stod(value_str);
            else if (key == "chim") params.chim = parse_double_vector(value_str); 
            else if (key == "ny") params.ny = std::stoi(value_str);
            else if (key == "nz") params.nz = std::stoi(value_str);
            else if (key == "p") params.p = std::stoi(value_str);
            else if (key == "sigma1") params.sigma1 = std::stod(value_str);
            else if (key == "Asp") params.Asp = std::stod(value_str);
            else if (key == "Y_range") params.Y_range = std::stod(value_str);
            else if (key == "Z_range") params.Z_range = std::stod(value_str);
            else if (key == "k_length") params.k_length = std::stoi(value_str);
            else if (key == "kstrt") params.kstrt = std::stod(value_str);
            else if (key == "kdiff") params.kdiff = std::stod(value_str);
            else if (key == "k1") params.k1 = parse_double_vector(value_str);
            else if (key == "BCzmag") params.BCzmag = std::stoi(value_str);
            else if (key == "BCzvel") params.BCzvel = std::stoi(value_str);
            else if (key == "BCymag") params.BCymag = std::stoi(value_str);
            else if (key == "BCyvel") params.BCyvel = std::stoi(value_str);
            else if (key == "mean_flow") params.mean_flow = std::stoi(value_str);
            else if (key == "B_profile") params.B_profile = std::stoi(value_str);
            else if (key == "Ra_start") params.Ra_start = std::stod(value_str);
            else if (key == "Ra_end_init") params.Ra_end_init = std::stod(value_str);
            else if (key == "Ra_extend_step") params.Ra_extend_step = std::stod(value_str);
            else if (key == "Ra_reduce_step") params.Ra_reduce_step = std::stod(value_str);
            else if (key == "Ra_search_limit") params.Ra_search_limit = std::stod(value_str);
            else if (key == "Ra_accuracy") params.Ra_accuracy = std::stod(value_str);
            else if (key == "outer_threads") params.outer_threads = std::stoi(value_str);
            else if (key == "inner_threads") params.inner_threads = std::stoi(value_str);
            // Add other parameters here if needed
            else {
                std::cerr << "Warning: Unknown parameter key '" << key << "' in " << filename
                          << " at line " << line_num << ". Skipping." << std::endl;
            }
        } catch (const std::invalid_argument& e) {
             std::cerr << "Error parsing value for key '" << key << "' in " << filename
                       << " at line " << line_num << ". Invalid format: " << value_str << std::endl;
             infile.close();
             return false;
        } catch (const std::out_of_range& e) {
             std::cerr << "Error parsing value for key '" << key << "' in " << filename
                       << " at line " << line_num << ". Value out of range: " << value_str << std::endl;
             infile.close();
             return false;
        } catch (const std::exception& e) { // Catch other potential errors (e.g., from vector parsing)
              std::cerr << "Error processing key '" << key << "' in " << filename
                       << " at line " << line_num << ": " << e.what() << std::endl;
              infile.close();
              return false;
        }
    }

    infile.close();
    params.calculate_derived(); // Calculate q, theta, and k1 (if not read explicitly)
    std::cout << "Finished reading parameters." << std::endl;
    return true;
}