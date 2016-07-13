function [TPR, ValidIndicesMatrix] = anomalyDataInterpretation3(dist_threshold, time_threshold, test_data, points_data, alerts_data, ano_data)
% Athanasios Tsiligkaridis, July 2nd 2016
%{
    INPUTS:  1) Distance threshold (in meters) 
             2) Time thresholds (in seconds)
    
    OUTPUTS: 1) True Positive Rate (percentage of indices whose minimum distances and time difference elements are below the inputted thresholds)
             2) Valid Indices Matrix (Indices whose minimum distances and time difference elements are below the inputted thresholds along with lat, long, and timestamp of the associated accident)

    DESCRIPTION: 
  
    Using a set of indices (defined in the program as a vector), we loop through all indices and focus on a specific one. 
    With this, we use the 'sample_path_jam_test_data_sorted.json' file to extract each index's uuid and timestamp info.
    With the timestamp, we look at the accident list (defined in the program) and we locate the accident(s) with the closest matching time values.  With this, we extract accident coordinates.
    With the uuid, we find its respective jam points from the 'points_dict_Mar_16_2016.mat' file. 
    With the jam points and the accident points, we create a distance matrix whos elements represent the distances in km between two coordinates.  
    For each index, we extract the minimum distance which represents the distance between the accident location and the closeste point in the jam.
    Also, we create a time difference element which is the difference between the accident timestamp and the index's time stamp.
    
    Finally, we take all of the preceding steps, replicate them for each
    index, and create a matrix of information where the first col is the
    index, the second is the min distance, and the last is the time
    difference.  

    With this, we then look for indices whose min distances and time
    differences are within user defined thresholds.  
    
    If for some index its elements are withing the respective thresholds, then that index was correctly identified. 
    The larger the thresholds the larger the TPR percentage output. 

    EXAMPLE:  

    [TPR,ValidIndicesMatrix] = anomalyDataInterpretation4(500,20*60)


 The table presented below has the following information in each column:  Refined Indices, Minimum Distances, Time Stamp Differences, Longitude of Corresponding Accident, Latitude of Corresponding Accident, and Accident Timestamp. 
%}

%clear, clc

% Initialization
num_closeneighbors = 10;

%% 1) Import test data from "sample_path_jam_test_data_sorted.json"
dat = loadjson(test_data);
testData = struct2cell(dat);
names = fieldnames(dat);
testIndices = zeros(1,length(names));
for i = 1:length(names)
    q = names{i};
    ind=strfind(q,'_');
    qq = q([(ind-1) (ind+1):end]);
    testIndices(i)= str2num(qq);
end

%% 2) Form refined indices vector. 

anoDict = load(ano_data);
FieldName = fieldnames(anoDict);

refinedIndices = zeros(length(FieldName), 1);
for i = 1:length(FieldName)
    refinedIndices(i) = extractfield(anoDict, FieldName{i});
end

%% 3) Import uuid point data from .json file -> 
a=load(points_data);
uuidPoints = struct2cell(a);

%% 4) Form data structure with information from each accident
% Get data from mat file.
A = load(alerts_data);
alert_information = struct2cell(A);
accidentInfo = cell(length(alert_information),1);
for x = 1:length(alert_information)
   accidentInfo{x} = {alert_information{x}.longitude,alert_information{x}.latitude,alert_information{x}.alertType,alert_information{x}.startTime,alert_information{x}.endTime}; 
end

%% 5) Compare timestamps and overall distances --> form vector of 1 and 0's for the final decisions.

%Calculates distance between 2 locations -> Distance is in km, x1000 for m.
anon_haversine=@(lat1,long1,lat2,long2)6372.8*(2.*asin(sqrt((sin(((pi/180)*(lat2-lat1))./2)).^2+cos((pi/180)*(lat1)).*cos((pi/180)*(lat2)).*(sin(((pi/180)*(long2-long1))./2)).^2)));  

% Use for table displays at the end
distances_min      = zeros(length(refinedIndices),1);
timediffs          = zeros(length(refinedIndices),1);
LAT_accident       = zeros(length(refinedIndices),1);
LONG_accident      = zeros(length(refinedIndices),1);
TimeStamp_accident = zeros(length(refinedIndices),1);

for i = 1:length(refinedIndices)
    % Index we currently look at
    testIndex = refinedIndices(i);

    % Using the index, retrieve its information
    a = find(testIndices == testIndex);
    
    % Extract data for a specific index
    info = testData{a};
    
    % Extract relevant information
    uuidInfo      = info.uuid;
    timestampInfo = info.startTime;
    
    % Using the uuid, extract points of a jam
    INDEX = 0;
    for ii = 1:length(uuidPoints)
        if (isequal(uuidInfo,uuidPoints{ii}.uuid))
            INDEX = ii;
        end
    end

    % Extract jam points for a certain uuid. 
    points = uuidPoints{INDEX}.pts;
    
    % Make latitude and longitude point vectors for a specific uuid/jam
    LAT  = points(:,:,2);
    LONG = points(:,:,1);
    
    
    % Using timestampInfo, we will get the coordinates of all
    % matching/close accidents.
    timeDifferences = zeros(1,length(accidentInfo));
    for iii = 1:length(accidentInfo)
        timeDifferences(iii) = timestampInfo - accidentInfo{iii}{4};
    end
    
    % Find elements whose differences are 0.
    set1 = find(timeDifferences == 0);
    
    % If the time stamp does not match any of the accident times, find the
    % closest accident times.
 
    % Get indices of closest neighbors
    if (isempty(set1))
        TD = abs(timeDifferences);
        
        [OUT,IND]=sort(TD);
        
        set1 = IND(1:num_closeneighbors);
        %disp('hi')
    end
    
    %
    %disp('hi')
    
    % Column1 -> lat, column2 -> long
    accidentPoints = zeros(length(set1),2);
    for iiii = 1:length(set1)
        ind_val = set1(iiii);
        
        % Using the index, we must extract the accident locations from the
        % accidentInfo cell array.
        accidentPoints(iiii,1) = str2num(accidentInfo{ind_val}{2}); 
        accidentPoints(iiii,2) = str2num(accidentInfo{ind_val}{1});
    end
    
    % Now we must compute distances. 
    distanceMat = zeros(length(LAT),length(set1));
    for j = 1:length(LAT)
        long_jam = LONG(j);
        lat_jam  = LAT(j);
        for k = 1:length(set1)
            long_accident = accidentPoints(k,2);
            lat_accident  = accidentPoints(k,1);
            distanceMat(j,k) = anon_haversine(lat_accident,long_accident,lat_jam,long_jam);
        end
    end
    
    % get minimum distance and column index of minimum value
    [distances_min(i),column] = min(min(distanceMat));
    
    % Using the column index, go into set1 and get the correct accident row
    % that corresponds to a minimum distance.
    AccidentRow = set1(column);
    
    % Using AccidentRow, extract the longitude, latitude, and timestamp of
    % the corresponding accident row.
    LAT_accident(i)       = str2num(accidentInfo{AccidentRow}{2});
    LONG_accident(i)      = str2num(accidentInfo{AccidentRow}{1});
    TimeStamp_accident(i) = accidentInfo{AccidentRow}{4};
    
    % Get time difference
    timediffs(i) = timeDifferences(set1(1));
end

%% Display Output -> Column layout:  indices, minimum distances, time differences, accident longitude, accident latitude, timestamp of accident.
format long g
fprintf('\n\n The table presented below has the following information in each column:  Refined Indices, Minimum Distances, Time Stamp Differences, Longitude of Corresponding Accident, Latitude of Corresponding Accident, and Accident Timestamp. \n\n');
finalmatrix = [refinedIndices,distances_min*1000,timediffs,LONG_accident,LAT_accident,TimeStamp_accident]
    
%% Obtain percentages of correctly found accidents from indices list.
counter_correct = 0;
 
validIndices = [];
for w = 1:length(timediffs)
    if ((finalmatrix(w,2)<dist_threshold) && (abs(finalmatrix(w,3))<time_threshold))
       validIndices = [validIndices finalmatrix(w,1)];
       counter_correct = counter_correct+1; 
    end
end

% Make valid indices matrix
ValidIndicesMatrix = zeros(length(validIndices),4);
for p = 1:length(validIndices)
    LOC = find(refinedIndices == validIndices(p));
    ValidIndicesMatrix(p,1) = validIndices(p);
    ValidIndicesMatrix(p,2) = finalmatrix(LOC,4); % Long
    ValidIndicesMatrix(p,3) = finalmatrix(LOC,5); % Lat
    ValidIndicesMatrix(p,4) = finalmatrix(LOC,6); % TimeStamp
end

% Calculate the percent of indices whose characteristics are within the set
% thresholds. 
percent_correct = counter_correct/length(timediffs);
percent_incorrect = 1-percent_correct;

TPR = percent_correct*100;
end
    
    
    
    
    
    
    
    
    
    
    
    
    
   


