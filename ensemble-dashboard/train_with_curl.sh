#!/bin/bash

# Simple training script using curl

echo "Training ensemble model with sample sessions..."

# Create a JSON payload with a few sample sessions
cat > train_payload.json << 'EOF'
{
  "sessions": [
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   16:57:41\n   SAV\n   MACHINE       0250\n   TRAN NO       227713\n   AUTHORIZATION 931521\n   ************7210\n   WITHDRAWAL    39000.00\n   AVAILABLE     1205.19\n   ACCOUNT         705.19\n   FROM SAVINGS\n   THANK YOU FOR USING\n   THE MULTILINK NETWORK",
    "CARD INSERTED\n    *7713*1*D*9,M-81,R-0\nCARD INITIALISE ATTEMPT = 1\n    *7713*1*D*9,M-81,R-0\nCARD INITIALISE ATTEMPT = 2\n    *7713*1*D*9,M-81,R-0\nCARD INITIALISE ATTEMPT = 3\n PIN ENTERED\n OPCODE = ABC\n  PAN 0006013*******7210\n  ---START OF TRANSACTION---",
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   16:58:35\n   MACHINE       0250\n   TRAN NO       227714\n   ***************7210\n   NOT ENOUGH FUNDS\n   FOR TRANSACTION\n   AVAILABLE\n   ACCOUNT\n         THANK YOU",
    "CARD INSERTED\n 17:02:32 ATR RECEIVED T=0\n OPCODE = FI\n  PAN 0004263********6440\n  ---START OF TRANSACTION---\n PIN ENTERED\n OPCODE = BBD\n 17:03:06 GENAC 1 : ARQC\n 17:03:08 GENAC 2 : TC\n NOTES STACKED\n CARD TAKEN",
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   17:03:07\n   SAV\n   MACHINE       0250\n   TRAN NO       227716\n   AUTHORIZATION 170316\n   ************6440\n   WITHDRAWAL    10000.00\n   ACCOUNT       22289.50\n   FROM SAVINGS\n         THANK YOU",
    "CARD INSERTED\n 17:06:33 ATR RECEIVED T=0\n OPCODE = FI\n  PAN 0004263********3397\n  ---START OF TRANSACTION---\n PIN ENTERED\n OPCODE = BBD\n 17:07:05 GENAC 1 : ARQC\n 17:07:07 GENAC 2 : TC\n NOTES STACKED\n CARD TAKEN",
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   17:07:07\n   SAV\n   MACHINE       0250\n   TRAN NO       227718\n   AUTHORIZATION 170418\n   ************3397\n   WITHDRAWAL    28000.00\n   ACCOUNT       334.14\n   FROM SAVINGS\n         THANK YOU",
    "CARD INSERTED\n 17:09:55 ATR RECEIVED T=0\n PIN ENTERED\n OPCODE = AB\n 17:10:14 GENAC 1 : ARQC\nEXTERNAL AUTHENTICATE: NO ARPC\n 17:10:16 GENAC 2 : AAC\n  PAN 0005357********5182\n  ---START OF TRANSACTION---",
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   17:10:16\n   MACHINE       0250\n   TRAN NO       227719\n   ***************5182\n   NOT ENOUGH FUNDS\n   FOR TRANSACTION\n         THANK YOU",
    "CARD INSERTED\n 17:13:37 ATR RECEIVED T=0\n OPCODE = FI\n  PAN 0004263********4691\n  ---START OF TRANSACTION---\n PIN ENTERED\n OPCODE = IB\n  PAN 0004263********4691\n  ---START OF TRANSACTION---",
    "CIM-DEPOSIT ACTIVATED\nCIM-SHUTTER OPENED\nCIM-ITEMS INSERTED\nOPERATION OK\nESC: 014\nJMD50-000,JMD100-000,\nJMD500-000,\nJMD1000-014,\nJMD2000-000,\nJMD5000-000\nVAL: 000\nREF: 000\nREJECTS:000",
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   17:14:28\n   MACHINE       0250\n   TRAN NO       227721\n   ***************4691\n   NO RESPONSE FROM\n   INSTITUTION\n         THANK YOU",
    "CARD INSERTED\n 17:18:52 ATR RECEIVED T=0\n OPCODE = FI\n  PAN 0004263********6533\n  ---START OF TRANSACTION---\n PIN ENTERED\n OPCODE = DAAC\n 17:19:21 GENAC 1 : ARQC\n 17:19:22 GENAC 2 : TC",
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   17:19:22\n   SAV  20415****\n   MACHINE       0250\n   TRAN NO       227723\n   AUTHORIZATION 171923\n   ************6533\n   BALANCE INQUIRY\n   AVAILABLE 192.74\n   ACCOUNT\n   FROM SAVINGS\n         THANK YOU",
    "CARD INSERTED\n 17:19:55 ATR RECEIVED T=0\n OPCODE = FI\n  PAN 0004263********3651\n  ---START OF TRANSACTION---\n PIN ENTERED\n OPCODE = BBD\n 17:20:22 GENAC 1 : ARQC\nEXTERNAL AUTHENTICATE: NO ARPC\n 17:20:23 GENAC 2 : AAC",
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   17:20:24\n   MACHINE       0250\n   TRAN NO       000000\n   ***************3651\n   INVALID AMOUNT\n      THANK YOU",
    "CARD INSERTED\n 17:21:18 ATR RECEIVED T=0\n OPCODE = FI\n  PAN 0004263********7087\n  ---START OF TRANSACTION---\n PIN ENTERED\n OPCODE = DAAC\n 17:21:50 GENAC 1 : ARQC\n 17:21:52 GENAC 2 : TC",
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   17:21:51\n   SAV  06438****\n   MACHINE       0250\n   TRAN NO       227726\n   AUTHORIZATION 172126\n   ************7087\n   BALANCE INQUIRY\n   AVAILABLE 7658.92\n   ACCOUNT\n   FROM SAVINGS\n         THANK YOU",
    "CARD INSERTED\n CARD TAKEN\n     *7726*1*D*0,M-090B0210B9,R-4\n     *7726*1*D*0,M-00,R-4\n TRANSACTION END\n     *PRIMARY CARD READER ACTIVATED*",
    "CARD INSERTED\n 17:22:29 ATR RECEIVED T=0\n OPCODE = FI\n  PAN 0004263********7087\n  ---START OF TRANSACTION---\n PIN ENTERED\n OPCODE = BBD\n 17:23:51 GENAC 1 : ARQC\nEXTERNAL AUTHENTICATE: NO ARPC\n 17:23:52 GENAC 2 : AAC",
    "N.C.B. MIDAS\n   NCB DUKE ST. BRANCH\n     DATE        TIME\n   2025/06/18   17:23:52\n   MACHINE       0250\n   TRAN NO       000000\n   ***************7087\n   INVALID AMOUNT\n      THANK YOU"
  ],
  "text_weight": 0.4,
  "statistical_weight": 0.3,
  "threshold": 0.5
}
EOF

echo "Sending training request..."
curl -X POST \
  -H "Content-Type: application/json" \
  -d @train_payload.json \
  http://localhost:8001/api/train

echo -e "\n\nChecking health status..."
curl http://localhost:8001/api/health

echo -e "\n\nTesting cluster sessions endpoint..."
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"cluster_id": 0}' \
  http://localhost:8001/api/cluster_sessions

echo -e "\n\nCleanup..."
rm train_payload.json

echo -e "\n\nTraining complete!"
