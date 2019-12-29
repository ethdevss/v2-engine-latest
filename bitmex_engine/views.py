from django.shortcuts import render
from django.http import HttpResponse
import requests

from marshmallow import Schema, fields

import datetime as dt
import numpy as np
import pandas as pd
import hashlib
import hmac
import json
import telegram
import time

from urllib.parse import urlparse, quote, urlencode
import urllib.parse

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.background import BackgroundScheduler

from mongoengine import *

strategy_scheduler = None
authorization_key = "qwboOcOHX2HDMaKY0iZf"

kei_id = "784845620"
sebin_id = "415000369"

class MarketData(Document):
    timestamp = DateTimeField(required=True, unique=True)
    symbol = StringField(required=True)
    open = IntField(required=True)
    high = IntField(required=True)
    low = IntField(required=True)
    close = IntField(required=True)
    trades = IntField(required=True)
    volume = IntField(required=True)
    meta = {'collection': 'candles'}


class TimeSchema(Schema):
    start_time = fields.DateTime()
    end_time = fields.DateTime()


class StrategyScheduler(object):
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        self.job_id = ''

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self.scheduler.shutdown()

    def kill_scheduler(self, job_id):
        try:
            self.scheduler_remove_job(job_id)
        except JobLookupError as err:
            print('Fail to stop scheduler')
            return

    def add_scheduler(self, type, job_id, func, seconds):
        if type == 'interval':
            self.scheduler.add_job(func, type, seconds=seconds, id=job_id)


class IndicatorGenerator(object):
    @classmethod
    def create_rsi_df(cls, close_list, period):
        df = pd.DataFrame(close_list, columns=['close'])
        rsi_period = period
        chg = df['close'].diff(1)

        gain = chg.mask(chg<0,0)
        df['gain'] = gain

        loss = chg.mask(chg>0,0)
        df['loss'] = loss

        avg_gain = gain.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(com=rsi_period-1, min_periods=rsi_period).mean()

        df['avg_gain'] = avg_gain
        df['avg_loss'] = avg_loss

        rs = abs(avg_gain / avg_loss)
        rsi = 100 - (100/(1+rs))

        df['rsi'] = rsi
        return df

    @classmethod
    def create_low_df(cls, low_list):
        df = pd.DataFrame(low_list, columns=['low'])
        return df

    @classmethod
    def create_close_df(cls, close_list):
        df = pd.DataFrame(close_list, columns=['close'])
        return df

    @classmethod
    def is_5m_candle_close(cls, minute, second):
        is_just = (second >= 2 and second <= 5)
        if minute == 5 and is_just:
            return True
        elif minute == 10 and is_just:
            return True
        elif minute == 15 and is_just:
            return True
        elif minute == 20 and is_just:
            return True
        elif minute == 25 and is_just:
            return True
        elif minute == 30 and is_just:
            return True
        elif minute == 35 and is_just:
            return True
        elif minute == 40 and is_just:
            return True
        elif minute == 45 and is_just:
            return True
        elif minute == 50 and is_just:
            return True
        elif minute == 55 and is_just:
            return True
        else:
            return False


class DivergenceStrategy(object):
    def __init__(self, target_rsi, order_quantity):
        self.name = "DivergenceStrategy"
        self.target_rsi = target_rsi
        self.order_quantity = order_quantity
        self.rsi_period = 14
        self.basis_candle = dict()
        self.trade_candle = dict()
        self.basis_result = False
        self.trade_result = False

    def run(self):
        current_timestamp = dt.datetime.now()
        minute = current_timestamp.minute
        second = current_timestamp.second

        # 현재 포지션을 확인한다.
        is_open = BitmexAPIToolKit.get_current_position("XBTUSD")

        if is_open:
            # 포지션 진입 후에 15분이 지났다면, 시장가로 매수 포지션 정리한다.
            if self.trade_candle['timestamp'] + dt.timedelta(minutes=15) < current_timestamp:
                BitmexAPIToolKit.marketprice_order("XBTUSD", "Sell", self.order_quantity)
                message = "포지션 진입 후 15분이 지나서 포지션을 정리합니다!"
                TelegramBot.send_message(kei_id, message)
                TelegramBot.send_message(sebin_id, message)
            # 방금 막 5분봉이 마감됐는지 확인
            elif IndicatorGenerator.is_5m_candle_close(minute, second):
                candles = MarketData.objects().order_by('-timestamp')[:4032]
        
                # 캔들 데이터 중 Close가격에 대한 정보를 리스트에 저장한다.
                close_list = [float(candle.close) for candle in candles]
       
                close_list = np.asarray(close_list)
                close_list = close_list[::-1]

                close_df = IndicatorGenerator.create_close_df(close_list)
                close_tail_df = close_df.tail(1)
                close_5m_price = close_tail_df['close'].values[0]

                # 진입 시 가격보다 만들어진 5분봉의 Close 가격이 더 낮을경우 손절한다.
                if self.trade_candle['entry_price'] > close_5m_price:
                    BitmexAPIToolKit.marketprice_order("XBTUSD", "Sell", self.order_quantity)
                    message = "포지션 진입 후 만들어진 5분봉의 Close 가격이 진입 시 가격보다 낮으므로 손절한다."
                    TelegramBot.send_message(kei_id, message)
                    TelegramBot.send_message(sebin_id, message)
            return    
        elif is_open == False and self.basis_candle and self.trade_candle:
            self.basis_candle = dict()
            self.trade_candle = dict()

        if IndicatorGenerator.is_5m_candle_close(minute, second):
            TelegramBot.send_message(kei_id, "5분봉 마감 직후")
            TelegramBot.send_message(sebin_id, "5분봉 마감 직후")
            # 기준봉이 구해져있는 상황이 아니라면, 기준봉만 구한다.
            if not self.basis_candle:
                self.basis_result = self.get_basis_candle()
            else:
                # 기준봉이 구해졌지만, 300분동안 거래봉을 구하지 못한 경우 기준봉은 다시 초기화된다.
                if self.basis_candle['timestamp'] + dt.timedelta(hours=5) < current_timestamp:
                    self.basis_candle = {}
                    self.trade_result = False
                elif self.is_keep_basis_candle():
                    self.trade_result = self.get_trade_candle()
                else:
                    self.trade_result = False
        
        if self.basis_candle and self.trade_candle:
            buy_result = BitmexAPIToolKit.marketprice_order("XBTUSD", "Buy", self.order_quantity)
            self.trade_candle['entry_price'] = buy_result['price']
       
        
    # 기존에는 5분봉이 마감되기전에 기준봉 여부를 판단했다.
    # 앞으로는 봉이 마감되고, 3초내로 기준봉 여부를 판단한다.
    # 그 이유는 봉이 마감되기전에 rsi를 구하려고 하다보니, 정확한 rsi 계산이 불가능했기 때문이다.
    # 정확한 RSI 계산을 하기 위해서는 봉이 마감된 순간의 정확한 Close 가격이 필요하다.
    def get_basis_candle(self):
        candles = MarketData.objects().order_by('-timestamp')[:4032]
        
        # 캔들 데이터 중 Close가격에 대한 정보를 리스트에 저장한다.
        close_list = [float(candle.close) for candle in candles]

        # 캔들 데이터 중 Low가격에 대한 정보를 리스트에 저장한다.
        low_list = [float(candle.low) for candle in candles]

        close_list = np.asarray(close_list)
        close_list = close_list[::-1]

        low_list = np.asarray(low_list)
        low_list = low_list[::-1]

        # historical candle close 가격을 기준으로 RSI를 구한다.
        # dataframe에는 과거 RSI부터 최신 RSI에 대한 정보가 저장된다.
        close_df_with_rsi = IndicatorGenerator.create_rsi_df(close_list, self.rsi_period)
        print(close_df_with_rsi)

        close_tail_df = close_df_with_rsi.tail(1)
        candle_5m_rsi = close_tail_df['rsi'].values[0]
        
        # low price에 대한 dataframe을 만든다.
        low_df = IndicatorGenerator.create_low_df(low_list)
        last_2_row = low_df.tail(2)

        # 전전 5분봉의 low 가격
        bb_low_price = last_2_row['low'].values[0]

        # 직전 5분봉의 low 가격
        b_low_price = last_2_row['low'].values[1]

        current_timestamp = dt.datetime.now()
 
        # 기준봉이 되기 위한 조건
        # 직전 5분봉과 전전 5분봉과의 Low 가격을 비교, 가격이 하락한 경우, RSI가 23미만인지 비교
        if bb_low_price > b_low_price and candle_5m_rsi < self.target_rsi:
            self.basis_candle['price'] = b_low_price
            self.basis_candle['rsi'] = candle_5m_rsi
            self.basis_candle['timestamp'] = current_timestamp
            message = "기준봉을 구했습니다! 기준봉의 price : " + str(b_low_price) + "\n" + \
                      " 기준봉의 rsi : " + str(candle_5m_rsi) + "\n" + \
                      " 기준봉의 timestamp : " + str(current_timestamp)
            TelegramBot.send_message(kei_id, message)
            TelegramBot.send_message(sebin_id, message)
            return True
        else:
            return False

    def is_keep_basis_candle(self):
        candles = MarketData.objects().order_by('-timestamp')[:4032]
        
        # 캔들 데이터 중 Close가격에 대한 정보를 리스트에 저장한다.
        close_list = [float(candle.close) for candle in candles]

        # 캔들 데이터 중 Low가격에 대한 정보를 리스트에 저장한다.
        low_list = [float(candle.low) for candle in candles]

        close_list = np.asarray(close_list)
        close_list = close_list[::-1]

        low_list = np.asarray(low_list)
        low_list = low_list[::-1]

        # historical candle close 가격을 기준으로 RSI를 구한다.
        # dataframe에는 과거 RSI부터 최신 RSI에 대한 정보가 저장된다.
        close_df_with_rsi = IndicatorGenerator.create_rsi_df(close_list, self.rsi_period)

        close_tail_df = close_df_with_rsi.tail(1)
        candle_5m_rsi = close_tail_df['rsi'].values[0]
        
        # low price에 대한 dataframe을 만든다.
        low_df = IndicatorGenerator.create_low_df(low_list)
        last_row = low_df.tail(1)
        current_timestamp = dt.datetime.now()

        if self.basis_candle['rsi'] > candle_5m_rsi and self.basis_candle['price'] <= last_row['low'].values[0]:
            self.basis_candle = {}
            message = "기준봉보다 RSI는 하락했지만, 가격은 하락하지 않았으므로 기준봉이 초기화 되었습니다."
            TelegramBot.send_message(kei_id, message)
            TelegramBot.send_message(sebin_id, message) 
            return False
        elif self.basis_candle['rsi'] > candle_5m_rsi and self.basis_candle['price'] > last_row['low'].values[0]:
            self.basis_candle['price'] = last_row['low'].values[0]
            self.basis_candle['rsi'] = candle_5m_rsi
            self.baiss_candle['timestamp'] = current_timestamp
            message = "기준봉보다 RSI가 하락하였고, 가격도 하락하였으므로 새로운 기준봉이 생겼습니다."
            TelegramBot.send_message(kei_id, message)
            TelegramBot.send_message(sebin_id, message)
            return False
        elif candle_5m_rsi > self.target_rsi and self.basis_candle['price'] > last_row['low'].values[0]:
            self.basis_candle = {}
            message = "RSI가 23(target_rsi)을 초과, 기준봉보다 가격이 하락해서 기준봉이 초기화 되었습니다."
            TelegramBot.send_message(kei_id, message)
            TelegramBot.send_message(sebin_id, message)
            return False
        else:
            return True

    def get_trade_candle(self):
        # 거래봉이 되기 위해서는 기준봉보다 rsi가 높아야한다.
        # 거래봉은 기준봉보다 가격이 하락했을때만 만들어진다.
        candles = MarketData.objects().order_by('-timestamp')[:4032]
        
        # 캔들 데이터 중 Close가격에 대한 정보를 리스트에 저장한다.
        close_list = [float(candle.close) for candle in candles]

        # 캔들 데이터 중 Low가격에 대한 정보를 리스트에 저장한다.
        low_list = [float(candle.low) for candle in candles]

        close_list = np.asarray(close_list)
        close_list = close_list[::-1]
 
        low_list = np.asarray(low_list)
        low_list = low_list[::-1]  

        # historical candle close 가격을 기준으로 RSI를 구한다.
        # dataframe에는 과거 RSI부터 최신 RSI에 대한 정보가 저장된다.
        close_df_with_rsi = IndicatorGenerator.create_rsi_df(close_list, self.rsi_period)

        close_tail_df = close_df_with_rsi.tail(1)
        candle_5m_rsi = close_tail_df['rsi'].values[0]

        # low price에 대한 dataframe을 만든다.
        low_df = IndicatorGenerator.create_low_df(low_list)
        last_row = low_df.tail(1)

        candle_5m_low_price = last_row['low'].values[0] 
        current_timestamp = dt.datetime.now()

        if self.basis_candle['price'] > candle_5m_low_price and self.basis_candle['rsi'] < candle_5m_rsi:
            self.trade_candle['price'] = candle_5m_low_price
            self.trade_candle['rsi'] = candle_5m_rsi
            self.trade_candle['timestamp'] = current_timestamp
            message = "거래봉을 구했습니다. 거래봉 price : " + str(candle_5m_low_proce) + "\n" + \
                      " 거래봉 rsi : " + str(candle_5m_rsi) + "\n" + \
                      " 거래봉 timestamp : " + str(current_timestamp)
            TelegramBot.send_message(kei_id, message)
            TelegramBot.send_message(sebin_id, message)
            return True
        else:
            return False

        
class BitmexAPIToolKit(object):
    bitmex_host = "https://www.bitmex.com"
    api_key = ""
    api_secret = ""

    @classmethod
    def marketprice_order(cls, symbol, side, quantity):
        host = cls.bitmex_host
        ordType = "Market"
        orderQty = quantity
        postBody = {"symbol": symbol, "side": side, "orderQty": quantity, "ordType": "Market"}
        postBody = json.dumps(postBody)
        path = '/api/v1/order'
        url = host + path
        headers = BitmexAPIToolKit.get_private_request_header("POST", path, postBody)
        r = requests.post(url, headers=headers, data=postBody)
        result_json = r.json()
        return result_json
  
    @classmethod 
    def get_current_position(cls, symbol):
        host = cls.bitmex_host
        filter = '{"symbol": "XBTUSD"}'
        filter = urllib.parse.quote_plus(filter)
        path = '/api/v1/position' + '?filter=' + filter
        url = host + path
        headers = BitmexAPIToolKit.get_private_request_header("GET", path, '')
        r = requests.get(url, headers=headers).json()
        if not r:
            return False
        else:
            is_open = result_json[0]['isOpen']
            return is_open

    @classmethod
    def generate_signature(cls, secret, verb, path, expires, postBody):
        message = bytes(verb + path + str(expires) + postBody, 'utf-8')
        signature = hmac.new(bytes(secret, 'utf-8'), message, digestmod=hashlib.sha256).hexdigest()
        return signature

    @classmethod
    def get_private_request_header(cls, verb, path, body):
        expires = int(time.time() + 3600)
        post_body = body
        if body == '':
            post_body = ''
        signature = BitmexAPIToolKit.generate_signature(cls.api_secret, verb, path, expires, post_body)
        headers = {
            'content-type': 'application/json',
            'Accept': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
            'api-expires': str(expires),
            'api-key': cls.api_key,
            'api-signature': signature
        }
        return headers

    @classmethod
    def get_current_price(cls):
        url = "https://www.bitmex.com/api/v1/trade?symbol=XBT&count=10&reverse=true"
        response = requests.get(url)
        last_traded_price = response.json()[0]['price']
        return last_traded_price


class TelegramBot(object):
    token = '985728867:AAE9kltQqpmIdwPi510h4fzfQas59besQzE'
    bot = telegram.Bot(token)

    @classmethod
    def send_message(cls, chat_id, message):
        cls.bot.send_message(chat_id=chat_id, text=message)
        

# Create your views here.

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def add_strategy(request):
    if request.headers['Authorization'] == authorization_key:
        global strategy_scheduler
        strategy = DivergenceStrategy(40, 500)
        strategy_scheduler.add_scheduler('interval', "2", strategy.run, 3)
        return HttpResponse("run strategy")
    else:
        return HttpResponse("Invalid Request")


def remove_strategy(request):
    if request.headers['Authorization'] == authorization_key:
        global strategy_scheduler
        strategy_scheduler.kill_scheduler("2")
        return HttpResponse("stop strategy")
    else:
        return HttpResponse("Invalid Request")


def init_engine(request):
    if request.headers['Authorization'] == authorization_key:
        global strategy_scheduler
        strategy_scheduler = StrategyScheduler()
        # connect to mongo db
        connection = connect(db='market_data')
        return HttpResponse("Initialize Trading Engine")
    else:
        return HttpResponse('Invalid Request')


def shutdown_engine(request):
    if request.headers['Authorization'] == authorization_key:
        global strategy_scheduler
        del strategy_scheduler
        return HttpResponse("Shutdown Trading Engine")
    else:
        return HttpResponse('Invalid Request')
