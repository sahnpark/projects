### Load packages

import numpy as np
import pandas as pd
import sys


### Load initial parameters

CITIES = { "1": "Chicago",
           "2": "New York City",
           "3": "Washington" }
INFO_TYPE = { "1": "popular times of travel",
              "2": "popular stations and trip",
              "3": "trip duration",
              "4": "user info" }
CITY_DATA = { 'chicago': 'chicago.csv',
              'new york city': 'new_york_city.csv',
              'washington': 'washington.csv' }

def get_response(response, expected, q_to_user, wheninvalid, whensuccess):
    """
    Gets expected response from the user.

    Input - response: predefined response.
                        This must not match with one of the expected response.
          - expected: in List (i.e. ['1', '2', '3'])
          - q_to_user: question to user in str (i.e. 'What is your favorite number?')
          - wheninvalid_input: error message for invalid input in str
          - whensuccess: success message in str
    """
    while True:
        try:
            while response not in expected:
                response = input(q_to_user).lower()
                if response not in expected:
                    print(wheninvalid)
                else:
                    print(whensuccess)
            break
        except KeyboardInterrupt:
            print('Oops! Terminating the program...')
            sys.exit("Error: KeyboardInterrupt. Please restart.")
        except Exception:
            print('\nError. Try again!\nIf you want to quit, press Ctrl + C then Enter.\n')
    return response

def sec_to_dur(sec):
    """
    Returns seconds to duration in string.
    Format: Days, Hours, Minutes, and Seconds
    """
    total_sec = round(sec % 60)
    total_sec_remain = int(sec / 60)
    total_min = total_sec_remain % 60
    total_min_remain = int(total_sec_remain / 60)
    total_hr = total_min_remain % 24
    total_day = int(total_min_remain / 24)
    dur = (str(total_day) + " Days " + str(total_hr) + " Hours "
           + str(total_min) + " Minutes " + str(total_sec) + " Seconds")
    return dur

def get_filters():
    """
    Asks user to specify a type information and a city to analyze.
    Inputs are taken from the user. Inital parameters are loaded with packages.

    Outputs: DESIRED_INFO - dictionary key of INFO_TYPE
             DESIRED_CITY - dictionary key of CITIES
    """
    DESIRED_INFO = 0
    expectedinfo = ['1', '2', '3', '4']
    qinfo = "\nWhat do you want to know?\n\
    1 = Popular times of travel \n\
    2 = Popular stations and trip \n\
    3 = Trip duration \n\
    4 = User info \n\
    Enter a number:     "
    invalinfo = '\nNot a valid input! Try again! Hint: Enter 1, 2, 3, or 4'
    successinfo = '\nOK.'
    DESIRED_INFO = get_response(DESIRED_INFO, expectedinfo, qinfo, invalinfo, successinfo)

    DESIRED_CITY = 0
    expectedcity = ['1', '2', '3']
    qcity = "\nWhich city? Response as a number. \n\
    1 = Chicago \n\
    2 = New York City \n\
    3 = Washington \n\
    Enter a number:     "
    invalcity = "\nNot a valid input! Try again! Hint: Enter 1, 2, or 3"
    successcity= " "
    DESIRED_CITY = get_response(DESIRED_CITY, expectedcity, qcity, invalcity, successcity)

    print('You want to know about {} in {}!\n'.format(INFO_TYPE[DESIRED_INFO], CITIES[DESIRED_CITY]))
    return DESIRED_INFO, DESIRED_CITY

def times_of_travel(df):
    """
    Prints common times of travel (i.e., occurs most often in the start time)
    """

    months_dic = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                  5: 'May', 6: 'June', 7: 'July', 8: 'August',
                  9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    weekdays_dic ={0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    df_month = pd.DatetimeIndex(df['Start Time']).month # find the most common month
    (values,counts) = np.unique(df_month,return_counts=True)
    ind=np.argwhere(counts==max(counts))
    mon_num = np.reshape(values[ind], (len(ind)),)
    mon_ans = "Most common month(s) of travel:  "
    for index in range(len(mon_num)):
        mon_ans = mon_ans + months_dic[mon_num[index]] + " "
    print(mon_ans)

    df_daywk = pd.DatetimeIndex(df['Start Time']).dayofweek # find the most common day of wk
    (values_daywk,counts_daywk) = np.unique(df_daywk,return_counts=True)
    ind_daywk=np.argwhere(counts_daywk==max(counts_daywk))
    daywk_num = np.reshape(values_daywk[ind_daywk], (len(ind_daywk)),)
    daywk_ans = "Most common day(s) of the week of travel:  "
    for index in range(len(daywk_num)):
        daywk_ans = daywk_ans + weekdays_dic[daywk_num[index]] + " "
    print(daywk_ans)

    df_hr = pd.DatetimeIndex(df['Start Time']).hour # find the most common hour of the day
    (values_hr,counts_hr) = np.unique(df_hr,return_counts=True)
    ind_hr=np.argwhere(counts_hr==max(counts_hr))
    hr_num = np.reshape(values_hr[ind_hr], (len(ind_hr)),)
    hr_ans = "Most common hour(s) of the day of travel:  "
    for index in range(len(hr_num)):
        if hr_num[index] > 12:
            hr_ans = hr_ans + str(hr_num[index]-12) + "PM "
        elif hr_num[index] == 12:
            hr_ans = hr_ans + str(hr_num[index]) + "PM "
        elif hr_num[index] == 0:
            hr_ans = hr_ans + str(hr_num[index]+12) + "AM"
        else:
            hr_ans = hr_ans + str(hr_num[index]) + "AM "
    print(hr_ans)

def stations_trip(df):
    startend = pd.DataFrame(df, columns= ['Start Station', 'End Station'])
    startend['Both'] = ( "Start - " + startend['Start Station'].astype(str)
                         + ", End - " + startend['End Station'])
    start = startend['Start Station'].value_counts() #start station.
    end = startend['End Station'].value_counts() #end station
    both = startend['Both'].value_counts() #trip between two stations
    print("Most popular start station:  ", start.index[0])
    print("Most popular end station:  ", end.index[0])
    print("Most popular trip from start to end:  ", both.index[0])

def trip_duration(df):
    duration = pd.DataFrame(df, columns= ['Trip Duration'])
    print("Total travel time:  ", sec_to_dur(int(duration.sum())))
    print("Average travel time:  ", sec_to_dur(float(duration.mean())))

def user_info(df, city):
    if city == "3": #Washington
        print("Counts of each user type    ")
        usertype = pd.DataFrame(df, columns= ['User Type']).fillna('Not-specified').values
        (type,counts) = np.unique(usertype,return_counts=True)
        for n in range(len(type)):
            typename = str(type[n])
            print("    ", typename, ":  ", str(counts[n]))
        print("Gender and birth year data are not available in Washington")

    else: #other cities
        # usertype = pd.DataFrame(df, columns= ['User Type']).value_counts(dropna = False)
        userinfo = pd.DataFrame(df, columns= ['User Type', 'Gender', 'Birth Year'])
        userinfo_no_nan = userinfo.fillna('Not-specified')
        # count user type
        usertype = userinfo_no_nan['User Type'].value_counts(dropna = False)
        print("Counts of each user type    ")
        for n in range(len(usertype)):
            typename = str(usertype.index[n])
            # if typename == 'nan':
            #     typename = 'Not-specified'
            print("    ", typename, ":  ", str(int(usertype[n])))

        #count user gender
        usergender = userinfo_no_nan['Gender'].value_counts(dropna = False)
        print("counts of each gender  ")
        for n in range(len(usergender)):
            gender = str(usergender.index[n])
            # if gender == 'nan':
            #     gender = 'Not-specified'
            print("    ", gender, ":  ", str(int(usergender[n])))

        # Birth year
        print("Earliest year of birth:  ", int(userinfo['Birth Year'].min()))
        print("Most recent year of birth:  ", int(userinfo['Birth Year'].max()))
        print("Most common year of birth:  ", userinfo['Birth Year'].value_counts().index[0])

def print_raw(df):
    """
    Prints raw data if the user says 'Yes'.
    Input - df : raw dataframe
    """
    response = 'no answer yet'
    expected = ['yes', 'no', 'y', 'n']
    q_to_user = "\nDo you want to see the first five row of the raw data? [ Yes / No ] :     "
    wheninvalid = "Invalid input. Enter 'Yes' or 'No'."
    whensuccess = " "
    response = get_response(response, expected, q_to_user, wheninvalid, whensuccess)
    print(response)
    if (response == 'yes'or response == 'y'):
        print(df.head())


def main():
    while True:

        desired_info, desired_city = get_filters()

        # load data
        df = pd.read_csv(CITY_DATA[CITIES[desired_city].lower()])
        print("Note: Data only include from January 2017 to June 2017.\n")

        if desired_info == "1":
            times_of_travel(df)
        elif desired_info == "2":
            stations_trip(df)
        elif desired_info == "3":
            trip_duration(df)
        elif desired_info == "4":
            user_info(df, desired_city)
        else:
            print('\nNot available!\n')

        print_raw(df)

        restart_r = 'no reply yet'
        restart_e = ['yes', 'no', 'y', 'n']
        restart_q = '\nWould you like to restart? [ Yes / No ] :     '
        restart_i = "Invalid input. Enter 'Yes' or 'No'."
        restart_s = " "
        restart = get_response(restart_r, restart_e, restart_q, restart_i, restart_s)
        if (restart == 'no'or restart == 'n'):
            print('Closing... Bye-bye!')
            break

if __name__ == "__main__":
	main()