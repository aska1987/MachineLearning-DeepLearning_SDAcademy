select * from student;
select * from professor;
select * from tab;
desc student;
desc professor;
desc salgrade;
select * from department;
select dname,deptno from department;
#���� ���̺��� ������ȣ, �����̸�, �޿��� ����ϼ���. 
select profno,name,sal from professor;

#�赵�� �������� �޿���?
select sal from professor where name='�赵��';

#�л� ���̺��� �й�,�̸�,Ű,������,�а���ȣ�� ����ϼ���.
select studno,name,height,weight,deptno from student;

select distinct deptno from student;
select deptno from student;
select distinct deptno,grade from student;

select dname dept_name, deptno as DN
from department;
select dname "Department Name", deptno "�μ���ȣ"
from department;

#�л����̺��� �й�,�̸�,�а���ȣ#�� ����ϼ���.
select studno "�й�",name "�̸�",deptno "�а���ȣ#" from student;

select studno || '��' || name "student"
from student;

select name, weight, weight*2.2 as weight_pound 
from student;

#�����̸�,�޿�,������ ���ϼ���. 
select name,sal,(sal+comm)*12 ���� 
from professor;

select rowid,studno,name,userid,deptno from student;

#�������̺��� 500�̻�����鼭 101���а� ������ �̸�,�а���ȣ,�޿� ���
select name,deptno,sal 
from professor 
where sal>=500 and deptno=101;

#�������̺��� ������ �α����� �ƴ� �������� �̸�,����,�а���ȣ,�޿��� ���
select name,position,deptno,sal
from professor
where not position='�α���';
#�������̺��� ������ �α����� �ƴϸ鼭 �޿��� 350���Ϸ� �޴� �������̸�,�а���ȣ,����,�޿��� ����ϼ���.
select name,deptno,position,sal
from professor
where not position='�α���' and sal<=350;

#1. ���� ���̺��� ������ �������� ����ϼ��� 
position
--------
����
�α���
������
���Ӱ���
select distinct position from professor;
#2. �Ʒ� ���Ǵ� ������ �����ϰ��ִ�. �°� �����ؼ� �����غ�����
select ename,job,sal*12 as yearly_sal
from emp;
select * from emp;
desc emp;
#3. ������̺��� �� ���̺��� employee and title �̰�, �޸��� �������� ���е�,�̸��� ������ ����ǵ��� ���
employee and title
------------------
smith,clerk
select last_name || ', '|| job_id
from employees;
select * from employees;
select * from jobs;
select * from emp;
#4. $2250 �̻��� ���� ����� �̸��� �޿� ,�μ���ȣ,���� ���
select last_name,salary,department_id,job_id
from employees
where salary>=2250;
#5. $2250 �̻��� �ް� ������ manager�� ����� �̸�,�޿�,�μ���ȣ,���� ���
select ename,sal,deptno,job
from emp
where sal>=2250 and job='MANAGER';

#��� ���̺��� �޿��� $1500~5000 �̰� ������ president �� analyst �� ��� ����� ���� ���,�̸�,����,�޿������
select empno,ename,job,sal
from emp
where sal>=1500 and sal<=5000 and job in('PRESIDENT','ANALYST');

