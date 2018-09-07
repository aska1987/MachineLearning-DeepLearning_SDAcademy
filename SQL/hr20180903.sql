select * from student;
select * from professor;
select * from tab;
desc student;
desc professor;
desc salgrade;
select * from department;
select dname,deptno from department;
#교수 테이블에서 교수번호, 교수이름, 급여를 출력하세요. 
select profno,name,sal from professor;

#김도훈 교수님의 급여는?
select sal from professor where name='김도훈';

#학생 테이블에서 학번,이름,키,몸무게,학과번호를 출력하세요.
select studno,name,height,weight,deptno from student;

select distinct deptno from student;
select deptno from student;
select distinct deptno,grade from student;

select dname dept_name, deptno as DN
from department;
select dname "Department Name", deptno "부서번호"
from department;

#학생테이블에서 학번,이름,학과번호#를 출력하세요.
select studno "학번",name "이름",deptno "학과번호#" from student;

select studno || 'ㅁ' || name "student"
from student;

select name, weight, weight*2.2 as weight_pound 
from student;

#교수이름,급여,연봉을 구하세요. 
select name,sal,(sal+comm)*12 연봉 
from professor;

select rowid,studno,name,userid,deptno from student;

#교수테이블에서 500이상받으면서 101번학과 교수님 이름,학과번호,급여 출력
select name,deptno,sal 
from professor 
where sal>=500 and deptno=101;

#교수테이블에서 직급이 부교수가 아닌 교수님의 이름,직급,학과번호,급여를 출력
select name,position,deptno,sal
from professor
where not position='부교수';
#교수테이블에서 직급이 부교수가 아니면서 급여를 350이하로 받는 교수님이름,학과번호,직급,급여를 출력하세요.
select name,deptno,position,sal
from professor
where not position='부교수' and sal<=350;

#1. 교수 테이블에서 유일한 직위들을 출력하세요 
position
--------
교수
부교수
조교수
전임강사
select distinct position from professor;
#2. 아래 질의는 오류를 포함하고있다. 맞게 수정해서 실행해보세요
select ename,job,sal*12 as yearly_sal
from emp;
select * from emp;
desc emp;
#3. 사원테이블에서 열 레이블이 employee and title 이고, 콤마와 공백으로 구분된,이름과 직무를 연결되도록 출력
employee and title
------------------
smith,clerk
select last_name || ', '|| job_id
from employees;
select * from employees;
select * from jobs;
select * from emp;
#4. $2250 이상을 버는 사원의 이름과 급여 ,부서번호,직급 출력
select last_name,salary,department_id,job_id
from employees
where salary>=2250;
#5. $2250 이상을 받고 직급이 manager인 사원의 이름,급여,부서번호,직급 출력
select ename,sal,deptno,job
from emp
where sal>=2250 and job='MANAGER';

#사원 테이블에서 급여가 $1500~5000 이고 직무가 president 나 analyst 인 모든 사원에 대해 사번,이름,직무,급여를출력
select empno,ename,job,sal
from emp
where sal>=1500 and sal<=5000 and job in('PRESIDENT','ANALYST');

