class AltitudeReward:
    def __init__(self,weight=1.0):
        self.weight=weight
    def __call__(self,env):
        alt_error=abs(env.drone.position[2]-env.target_altitude)#z coordinate - desired altitude=error for us
        return (max(0.0,1.0-(0.1*alt_error))*self.weight)
