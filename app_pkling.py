import os
import numpy as np
import pandas as pd
from king_classic_pkling import PlayGolf, Player
from flask import Flask, request, redirect, url_for, render_template
from collections import Counter
from os import listdir
from os.path import isfile, join
import pickle
import pdb


app = Flask(__name__)
golf = PlayGolf()


# helper function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extenstions


# home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# general info page
@app.route('/general_info', methods=['GET'])
def general_info():
    return render_template('general_info.html')


# locations map
@app.route('/past_locations', methods=['GET'])
def past_locations():
    return render_template('past_locations.html')


# past results page
@app.route('/past_results', methods=['GET'])
def past_results():
    return render_template('past_results.html')


# add a player page
@app.route('/add_player', methods=['GET', 'POST'])
def add_player():
    if request.method == 'POST':
        f_name = request.form['first_name'].capitalize()
        l_name = request.form['last_name'].capitalize()
        hdcp = float(request.form['hdcp'])
        tee1 = request.form['tee1']
        tee2 = request.form['tee2']
        tee3 = request.form['tee3']
        tee4 = request.form['tee4']
        tee5 = request.form['tee5']
        tee6 = request.form['tee6']

        tees = {
            'Pine Needles': tee1,
            'Pine Needles Replay': tee2,
            'Southern Pines': tee3,
            'Southern Pines Replay': tee4,
            'Mid Pines': tee5,
            'Mid Pines Replay': tee6
        }
        # skins = request.form['in_skins']
        # if skins == 'True':
        #     skins = True
        # else:
        #     skins = False
        full_name = f_name.strip() + ' ' + l_name.strip()
        full_name = full_name

        golf.add_player(full_name, hdcp, tees)
        msg = 'Player added successfully'
        return render_template('add_player.html', msg=msg)

    return render_template('add_player.html')


# enter scores page
@app.route('/enter_scores', methods=['GET', 'POST'])
def enter_scores():
    allfiles = [f for f in listdir(golf.pkl_path) if isfile(join(golf.pkl_path, f))]
    golfers = []
    for pf in allfiles:
        with open('{}'.format(golf.pkl_path) + pf, 'rb') as f:
            golfers.append(pickle.load(f))

    players = [golfer.name for golfer in golfers]
    players.sort()

    holes = [x for x in range(1,19)]
    scores = [x for x in range(0,14)]
    course = 'None'

    if request.method == 'POST':
        try:
            course = request.form['course']
            on_course = course
            hole = int(request.form['hole'])
            next_hole = hole + 1

            p1 = request.form['player1']
            p2 = request.form['player2']
            p3 = request.form['player3']
            p4 = request.form['player4']

            golfers = [p1, p2, p3, p4]
            # golfers = [request.form['player1'], request.form['player2'], request.form['player3'], request.form['player4']]
            golfers = [golfer for golfer in golfers if golfer != 'None']

            g_scores = [request.form['score1'], request.form['score2'], request.form['score3'], request.form['score4']]
            g_scores = [int(score) for score in g_scores if score != 'None']

            if course == 'None' or hole == 0 or not g_scores or not golfers or len(golfers) != len(g_scores):
                msg = 'An error occured. Please ensure a course, hole, and at least one golfer and score are selected.'
                return render_template('enter_scores.html', players=players, holes=holes, scores=scores, msg=msg, on_course=course, next_hole=hole, p1=p1, p2=p2, p3=p3, p4=p4)

            if len([x for x, count in Counter(golfers).items() if count > 1]) >= 1:
                msg = 'The same golfer was selected twice. Please try again.'
                return render_template('enter_scores.html', players=players, holes=holes, scores=scores, msg=msg, on_course=course, next_hole=hole, p1=p1, p2=p2, p3=p3, p4=p4)

            gns = list(zip(golfers, g_scores))
            for golfer, score in gns:
                golf.add_score(golfer, course, hole, score)

            scorecard_df = golf.player_scorecards(golfers, course)
            msg = 'Scores entered successfully!'

            if next_hole > 18:
                on_course = 'None'
                next_hole = 0
                p1 = 'None'
                p2 = 'None'
                p3 = 'None'
                p4 = 'None'

            return render_template('enter_scores.html', players=players, holes=holes, scores=scores, msg=msg, scorecard_df=scorecard_df.to_html(), course=course, on_course=on_course, next_hole=next_hole, p1=p1, p2=p2, p3=p3, p4=p4)
        except:
            course = request.form['course']
            hole = int(request.form['hole'])
            p1 = request.form['player1']
            p2 = request.form['player2']
            p3 = request.form['player3']
            p4 = request.form['player4']
            msg = 'An error occurred. Please try again.'
            return render_template('enter_scores.html', players=players, holes=holes, scores=scores, msg=msg, on_course=course, next_hole=hole, p1=p1, p2=p2, p3=p3, p4=p4)
    else:
        return render_template('enter_scores.html', players=players, holes=holes, scores=scores, on_course='None', next_hole=0, p1='None', p2='None', p3='None', p4='None')


# leaderboard page
@app.route('/leaderboard', methods=['GET', 'POST'])
def leaderboard():
    leaderboard_df = golf.leaderboard()
    return render_template('leaderboard.html', leaderboard_df=leaderboard_df.to_html(index=False))


# skins page
@app.route('/skins', methods=['GET', 'POST'])
def skins():
    if request.method == 'POST':
        try:
            course = request.form['skins_course']
            if course == 'None':
                msg = 'Please select a course.'
                return render_template('skins.html', msg=msg)

            skins_df = golf.calc_skins(course)
            golfers = skins_df['Player'].tolist()
            scorecard_df = golf.player_scorecards(golfers, course, skins=True)

            return render_template('skins_results.html', skins_df=skins_df.to_html(index=False), scorecard_df=scorecard_df.to_html(), course=course)

        except:
            msg = 'No skins were won.'
            return render_template('skins.html', msg=msg)

    return render_template('skins.html')


# scorecard page
@app.route('/scorecard', methods=['GET', 'POST'])
def scorecard():
    allfiles = [f for f in listdir(golf.pkl_path) if isfile(join(golf.pkl_path, f))]
    golfers = []
    for pf in allfiles:
        with open('{}'.format(golf.pkl_path) + pf, 'rb') as f:
            golfers.append(pickle.load(f))

    players = [golfer.name for golfer in golfers]
    players.sort()

    if request.method == 'POST':
        course = request.form['scorecard_course']
        net = request.form['net']
        if net == 'Yes':
            net = True
        else:
            net = False
        if course == 'None':
            msg = 'Please select a course.'
            return render_template('scorecard.html', players=players, msg=msg)

        golfers = request.form.getlist('golfers')
        scorecard_df = golf.player_scorecards(golfers, course, net)
        return render_template('scorecard_results.html', scorecard_df=scorecard_df.to_html(), course=course)

    return render_template('scorecard.html', players=players)


# teams page
@app.route('/teams', methods=['GET', 'POST'])
def teams():
    allfiles = [f for f in listdir(golf.pkl_path) if isfile(join(golf.pkl_path, f))]
    golfers = []
    for pf in allfiles:
        with open('{}'.format(golf.pkl_path) + pf, 'rb') as f:
            golfers.append(pickle.load(f))

    players = [golfer.name for golfer in golfers]
    players.sort()

    if request.method == 'POST':
        course = request.form['course']
        if course == 'None':
            msg = 'Please select a course.'
            return render_template('teams.html', players=players, msg=msg)

        p1 = [request.form['t1p1'],
                request.form['t2p1'],
                request.form['t3p1'],
                request.form['t4p1']]

        p2 = [request.form['t1p2'],
                request.form['t2p2'],
                request.form['t3p2'],
                request.form['t4p2']]

        p1 = [golfer for golfer in p1 if golfer != 'None']
        p2 = [golfer for golfer in p2 if golfer != 'None']

        if len(p1) != len(p2):
            msg = 'Teams not properly defined. Please try again.'
            return render_template('teams.html', players=players, msg=msg)

        golfers = list(zip(p1,p2))
        teams_df = golf.calc_teams(golfers, course)
        return render_template('teams_results.html', teams_df=teams_df.to_html(index=False), course=course)

    return render_template('teams.html', players=players)


# tees and handicaps page
@app.route('/tees_handicaps', methods=['GET', 'POST'])
def tees_handicaps():
    if request.method == 'POST':
        course = request.form['hdcp_course']
        if course == 'None':
            msg = 'Please select a course.'
            return render_template('tees_handicaps.html', msg=msg)

        hdcps_df = golf.show_handicaps(course)
        return render_template('tees_handicaps.html', hdcps_df=hdcps_df.to_html(index=False), course=course)

    return render_template('tees_handicaps.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
