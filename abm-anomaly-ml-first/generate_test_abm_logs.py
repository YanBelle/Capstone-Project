#!/usr/bin/env python3
"""Generate test ABM EJ logs with various anomaly patterns"""

import random
from datetime import datetime, timedelta

def generate_test_logs(filename="test_abm_logs.txt", num_sessions=100):
    """Generate test ABM logs with anomalies from the requirements"""
    
    with open(filename, 'w') as f:
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(num_sessions):
            session_time = base_time + timedelta(hours=i)
            
            # Normal transaction template
            transaction = f"""*{i:03d}*{session_time.strftime('%m/%d/%Y')}*{session_time.strftime('%H:%M')}*
     *TRANSACTION START*
CARD INSERTED
{session_time.strftime('%H:%M:%S')} ATR RECEIVED T=0
{(session_time + timedelta(seconds=5)).strftime('%H:%M:%S')} PIN ENTERED
{(session_time + timedelta(seconds=10)).strftime('%H:%M:%S')} OPCODE = BBD
GENAC 1 : ARQC
GENAC 2 : TC

  PAN 0004263********{random.randint(1000, 9999)}
  ---START OF TRANSACTION---
  
       N.C.B. MIDAS
   NCB RUBIS BORDER AVE
     DATE        TIME
   {session_time.strftime('%Y/%m/%d')}   {session_time.strftime('%H:%M:%S')}
   MACHINE       0163
   TRAN NO       {100000 + i}
"""
            
            # Add anomalies based on examples
            if i % 20 == 0:  # Unable to dispense
                transaction += "   UNABLE TO DISPENSE\n"
            elif i % 25 == 1:  # Supervisor mode after transaction
                transaction += f"{(session_time + timedelta(seconds=20)).strftime('%H:%M:%S')} TRANSACTION END\n"
                transaction += "SUPERVISOR MODE ENTRY\nSUPERVISOR MODE EXIT\n"
                continue
            elif i % 30 == 2:  # Power reset after transaction
                transaction += f"{(session_time + timedelta(seconds=20)).strftime('%H:%M:%S')} TRANSACTION END\n"
                transaction += "[05pPOWER-UP/RESET\nAPTRA ADVANCE NDC 05.01.00[00p\n"
                continue
            elif i % 35 == 3:  # Cash retract error
                transaction += """A/C
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000
CASHIN RETRACT STARTED - RETRACT BIN
"""
            elif i % 40 == 4:  # Long delay in note taking
                transaction += f"{(session_time + timedelta(seconds=20)).strftime('%H:%M:%S')} NOTES PRESENTED 0,0,0,6\n"
                transaction += f"{(session_time + timedelta(seconds=35)).strftime('%H:%M:%S')} NOTES TAKEN\n"
            else:  # Normal completion
                transaction += f"   WITHDRAWAL    {random.choice([50000, 100000, 200000])}.00\n"
                transaction += "   FROM CHEQUING\n"
            
            transaction += "         THANK YOU\n"
            transaction += f"{(session_time + timedelta(seconds=25)).strftime('%H:%M:%S')} CARD TAKEN\n"
            transaction += f"{(session_time + timedelta(seconds=30)).strftime('%H:%M:%S')} TRANSACTION END\n"
            
            f.write(transaction)
            f.write("\n")
    
    print(f"Generated {num_sessions} test sessions in {filename}")
    print("Anomaly types included:")
    print("- Unable to dispense")
    print("- Supervisor mode after transaction")
    print("- Power reset after transaction")
    print("- Cash retract with device error")
    print("- Long delay between notes presented and taken")

if __name__ == "__main__":
    generate_test_logs()
