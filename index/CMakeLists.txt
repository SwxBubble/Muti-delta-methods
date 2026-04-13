add_library(index STATIC super_feature_index.cpp palantir_index.cpp
                            best_fit_index.cpp palantir_index_2.cpp
                            palantir_index_3.cpp
                            palantir_index_3.cpp)
target_link_libraries(index glog gflags feature)